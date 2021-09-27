import sys
sys.path.append("../")
from lumitexel_related import visualize_cube_slice,visualize_cube_slice_init,get_cube_slice_visualize_idxs
import numpy as np
import cv2

sample_num_edge = 8
brdf_not_slice_size = sample_num_edge*sample_num_edge*6
block_size = 32
UITLS_PATH = "../"
if __name__ == "__main__":
    visualize_cube_slice_init(UITLS_PATH)
    idxes = get_cube_slice_visualize_idxs()#[6144,2]    

    itr_count = 0

    block_num = sample_num_edge//block_size

    idx_collector = []
    idx_invert_collector = np.zeros(brdf_not_slice_size,np.int32)

    for x in range(block_num*4):
        for y in range(block_num*3):
            real_x = x*block_size
            real_y = y*block_size
            hit_point = (idxes[:, 0] == real_x) & (idxes[:, 1] == real_y)
            if np.count_nonzero(hit_point) != 0:
                for i in range(block_size):
                    for j in range(block_size):
                        hit_idx = (idxes[:, 0] == real_x+i) & (idxes[:, 1] == real_y+j)
                        idx = np.where(hit_idx)[0][0]
                        idx_collector.append(idx)
                        idx_invert_collector[idx] = len(idx_collector)-1
                itr_count+=1
    idx_collector = np.asarray(idx_collector,np.int32)
    idx_collector.tofile("./reorder_brdf_not_slice_to_{}x{}.bin".format(block_size,block_size))

    idx_invert_collector.tofile("./reorder_brdf_not_slice_to_{}x{}_revert.bin".format(block_size,block_size))

    print("itr count:",itr_count)


    data = np.fromfile("./test_brdf_not_slice_data.bin",np.float32).reshape([-1,brdf_not_slice_size])
    test_lumi = data[0]
    img_origin = visualize_cube_slice(test_lumi)
    reordered_lumi = data[:,idx_collector]
    reordered_lumi = reordered_lumi[0]
    inverted_lumi = reordered_lumi[idx_invert_collector]
    img_inverted = visualize_cube_slice(inverted_lumi)

    # shrinked_lumi = np.repeat(np.mean(reordered_lumi.reshape([-1,block_size*block_size]),axis=1,keepdims=True),block_size*block_size,axis=1).reshape([-1])
    # inverted_shrinked_lumi = shrinked_lumi[idx_invert_collector]
    # img_inverted_shrinked = visualize_cube_slice(inverted_shrinked_lumi)

    # deprecated_img = expand_img(shrink(img_origin,8,method="sum"),8,back_zero=True)

    cv2.imshow("img_origin",img_origin)
    cv2.imshow("img_inverted",img_inverted)
    # cv2.imshow("img_inverted_shrinked",img_inverted_shrinked)
    # cv2.imwrite("img_inverted_shrinked.png",img_inverted_shrinked*255)
    # cv2.imshow("deprecated_img",deprecated_img)
    cv2.waitKey(0)