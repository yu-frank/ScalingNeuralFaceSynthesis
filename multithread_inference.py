from dataset.warp_uv_dataset import WarpUVDataset
import torch
from torch.utils.data import DataLoader
from util import remove_t1_data, save_video
from multithreaded.multi_runner import InferenceRunner
import torch.multiprocessing as mp
import time

# Define dataset stats for RGB color here
stats = "0.4974, 0.4476, 0.4263, 0.1860, 0.1776, 0.1798"
stats = stats.split(',')
stats = [float(x) for x in stats]
mean = stats[:3]
std = stats[3:]

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def initialize(device):
    checkpoint_name = 'best_val_psnr.tar'
    model_path = './evaluate/pretrained_model'
    inference_runner = InferenceRunner(model_path, checkpoint_name=checkpoint_name, device=device, mean=mean, std=std)

    return inference_runner, None

def extractor_worker(wid, in_queue, cache_que, done_queue):
    print("worker", wid, "starting")
    # load the network, move things to the right GPU
    num_gpu = torch.cuda.device_count()
    gpu_id = wid%num_gpu 
    
    print("worker", wid, "using GPU ",gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id))

    # save a dummy background in memory
    background = torch.FloatTensor(torch.ones([1,3,512,512])).to(device)
    
    # load network weights etc.
    inference_runner, _ = initialize(device=device)
    
    inference_runner.model.eval()
    with torch.no_grad(): # note, this is no_grad thread-local
        while(True):
            # wait for new input frame
            do_cache, pose = in_queue.get()
            if do_cache == -1:
                return
            pose = inference_runner.move2GPU(pose)
            # switch between warp and caching
            if do_cache: # cache
                inference_runner.generator(pose)
                cache_que.put(wid)# just return the id to signal which thread is done
            else:
                out_img, mask  = inference_runner.warp(pose)
                out_img = inference_runner.generate_rgb_img_cuda(out_img, mask, background)
                done_queue.put(out_img)

    print("worker",wid,"Ending",'1')


if __name__ == '__main__':
    # Load Dataset
    data_dir = './data/'

    dataset = WarpUVDataset(data_dir, H=0, W=0, split='ear-adjust-test', view_direction=1,\
                            use_aug=False, use_mask=True, t_diff=1,
                            mean=mean, std=std, use_pose=True, normSplit='ear-adjust-train', 
                            use_exp=True, float_image=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    loaded_data = []
    for n, data in enumerate(dataloader):
        print(n)
        loaded_data.append(remove_t1_data(data))
        if n == 600:
            break
            
    print('loaded data length: ', len(loaded_data))
        
    num_threads = 2 # should be 2 (1 for debugging)
    warps_per_cache = 1 # should be 1 

    all_images = []

    pose_queues = [mp.Queue() for wid in range(num_threads)]
    out_queue = mp.Queue()
    cache_completed_queue = mp.Queue()

    producers = []
    # start threads
    for wid in range(num_threads):
        process = mp.Process(target=extractor_worker,
                             args=(wid, pose_queues[wid], cache_completed_queue, out_queue))
        producers.append(process)
        process.start()
        print("Started thread ",wid)

    # first round of caching
    warp_thread = 0
    cache_thread = 1 # should be 1 (0 for debugging on 1 GPU)

    print('done initialization')
    print("Start caching first frame")

    # read in first data:
    data = loaded_data[0]
    pose_queues[cache_thread].put((True,data)) # cache the first result
    count = 0
    start_time = None
    warmup = 6
    print('starting runtime')
    for i, data in enumerate(loaded_data[1:]):
        print("i: ", i)
        if i >= warmup:
            count += 1
            if i == warmup:
                start_time = time.time()

        if count % warps_per_cache == 0:
            cache_completed_queue.get(block=True)
            warp_thread = (warp_thread+1) % num_threads
            cache_thread = (cache_thread+1) % num_threads
            pose_queues[cache_thread].put((True,data,))

        pose_queues[warp_thread].put((False,data,))
        out_img = out_queue.get()

        all_images.append(out_img[0])
        

    print('len all images: ', len(all_images))
    print('count')
    print((count) / (time.time() - start_time))
    save_video('./result.mp4', all_images, 30)
    # stop workers
    for wid in range(num_threads):
        pose_queues[wid].put((-1,data)) # cache the first result

    

