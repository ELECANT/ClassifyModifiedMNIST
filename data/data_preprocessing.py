# "mnist_train_data" is the data file which contains a 60000*45*45 matrix(data_num*fig_w*fig_w)
# "mnist_train_label" is the label file which contains a 60000*1 matrix. Each element i is a number in [0,9]. 
# The dataset is saved as binary files and should be read by Byte. Here is an example of input the dataset and save a random figure.

import numpy as np
from PIL import Image
import cv2 

def example():
    data_num = 60000 #The number of figures
    fig_w = 45       #width of each figure

    data = np.fromfile("mnist_train_data", dtype=np.uint8)
    label = np.fromfile("mnist_train_label", dtype=np.uint8)

    print(data.shape)
    print(label.shape)

    #reshape the matrix
    data = data.reshape(data_num,fig_w,fig_w)
    print(data[2][20])
    print("After reshape:",data.shape)

    #choose a random index
    ind = np.random.randint(0,data_num)

    #print the index and label
    print("index: ",ind)
    print("label: ",label[ind])

    #save the figure
    im = Image.fromarray(data[ind])
    im.save("example.png")


def BFS(img, x, y, marked, marker):
    dxl = [-1, 0, 1, 0, -1, 0, 1, 0]
    dyl = [-1, 1, 1, -1, 0, 0, 0, 0]
    m, n = 45, 45
    head, tail = 0, 0 
    queue = [] 
    queue.append((x, y)) 
    marked[x][y] = marker 
    while head <= tail:
        firx, firy = queue[head]
        head += 1 
        for dx, dy in zip(dxl, dyl):
            tmpx = dx+firx
            tmpy = dy+firy 
            if tmpx < 0 or tmpx >= m or tmpy < 0 or tmpy >=n:
                continue
            if img[tmpx][tmpy] > 0 and marked[tmpx][tmpy] == 0:
                marked[tmpx][tmpy] = marker 
                queue.append((tmpx, tmpy)) 
                tail += 1 
    return tail+1 


def mask(img, thresh):
    n = 45
    marker = 0
    marked = []
    size_dict = dict() 

    for x in range(0, n):
        marked.append([])
        for y in range(0, n):
            marked[x].append(0)

    for x in range(0, n):
        for y in range(0, n):
            if marked[x][y] == 0 and img[x][y] > 0:
                marker += 1
                cnt = BFS(img, x, y, marked, marker) 
                size_dict[marker] = cnt 

    zero_list = set()
    for i in size_dict:
        if size_dict[i] < thresh:
            zero_list.add(i)

    for x in range(0, n):
        for y in range(0, n):
            if marked[x][y] in zero_list:
                img[x][y] = 0 
    return img 


def resize_img(train_file, test_file, raw_size, goal_size, train_save, test_save):
    train_data = np.load(train_file).reshape(60000, raw_size, raw_size)
    test_data = np.load(test_file).reshape(10000, raw_size, raw_size)

    new_train_data = np.zeros(shape=(60000, goal_size, goal_size), dtype=np.uint8)
    new_test_data = np.zeros(shape=(10000, goal_size, goal_size), dtype=np.uint8)
    for i in range(10000):
        if i % 1000 == 0: print('test', i)
        new_test_data[i] = cv2.resize(test_data[i], (goal_size, goal_size))
    np.save(test_save, new_test_data)
    for i in range(60000):
        if i % 1000 == 0: print('train', i)
        new_train_data[i] = cv2.resize(train_data[i], (goal_size, goal_size))
    np.save(train_save, new_train_data)



def main():
    train_data = np.load("mnist_train_raw.npy")
    test_data = np.fromfile("mnist_test_raw.npy")
    train_data = train_data.reshape(60000, 45, 45)
    test_data = test_data.reshape(10000, 45, 45)
    for i in range(10000):
        if i % 1000 == 0: print('test ', i)
        test_data[i] = mask(test_data[i], 20)
    test_data = test_data.reshape(10000, 45*45)
    np.save('mnist_test_clean.npy', test_data)

    for i in range(60000):
        if i % 1000 == 0: print('train ', i)
        train_data[i] = mask(train_data[i], 20)
    train_data = train_data.reshape(60000, 45*45)
    np.save('mnist_train_clean.npy', train_data)



def cut_img(img):
    high, low, left, right = -1, -1, -1, -1 
    for i in range(0, 45, 1):
        sumup = np.sum(img[i])
        if sumup > 0:
            high = i 
            break 
    for i in range(0, 45, 1):
        sumup = np.sum(img[:, i])    
        if sumup > 0:
            left = i 
            break 
    for i in range(44, -1, -1):
        sumup = np.sum(img[:, i])
        if sumup > 0:
            right = i 
            break 
    for i in range(44, -1, -1):
        sumup = np.sum(img[i])
        if sumup > 0: 
            low = i 
            break 

    hit = img[high:low+1, left:right+1]
    height = low+1-high 
    width = right+1-left
    if height > width: 
        new_h, new_w = 24, int(width*24./height)
    elif height <= width: 
        new_h, new_w = int(height*24./width), 24
    rhit = cv2.resize(hit, (new_w, new_h))
    lefttop_x = int((28-new_w)/2) 
    lefttop_y = int((28-new_h)/2)
    ret = np.zeros(shape=(28, 28), dtype=np.uint8)
    ret[lefttop_y:lefttop_y+new_h, lefttop_x:lefttop_x+new_w] = rhit
    
    return ret 

    

def cut(train_file, test_file):
    train_data = np.load(train_file).reshape(60000, 45, 45)
    test_data = np.load(test_file).reshape(10000, 45, 45)

    new_train_data = np.zeros(shape=(60000, 28, 28), dtype=np.uint8)
    new_test_data = np.zeros(shape=(10000, 28, 28), dtype=np.uint8)
    
    for i in range(10000):
        if i % 1000 == 0: print('test', i)
        new_test_data[i] = cut_img(test_data[i])
    np.save('mnist_test_cut.npy', new_test_data)
    
    for i in range(60000):
        if i % 1000 == 0: print('train', i)
        new_train_data[i] = cut_img(train_data[i])
    np.save('mnist_train_cut.npy', new_train_data)
    


def show_dataset(num):
    from random import sample 
    samples = sample([i for i in range(0, 10000)], num*num)
    print(samples)

    def one_dataset(num, samples, file, m, save_name):
        data = np.load(file).reshape(-1, m, m)
        allimg = np.zeros(shape=(num*m, num*m), dtype=np.uint8)
        for i,j in enumerate(samples):
            row, col = int(i/num), int(i%num)
            # print(allimg[row*m:(row+1)*m, col*m:(col+1)*m].shape)
            # print(data[i,:,:].shape)
            allimg[row*m:(row+1)*m, col*m:(col+1)*m] = data[j,:,:]

        cv2.imwrite(save_name, allimg)

    one_dataset(num, samples, 'mnist_test_raw.npy', 45, 'display_raw.png')
    one_dataset(num, samples, 'mnist_test_clean.npy', 45, 'display_clean.png')
    one_dataset(num, samples, 'mnist_test_cut.npy', 45, 'display_cut.png')



if __name__ == '__main__':
    # main()   
    
    # cut('mnist_train_clean.npy', 'mnist_test_clean.npy')
    
    # resize_img('../data/mnist_train_cut_28.npy', '../data/mnist_test_cut_28.npy', 
    #            28, 45, 
    #            '../data/mnist_train_cut.npy', '../data/mnist_test_cut.npy')

    show_dataset(5)
    

