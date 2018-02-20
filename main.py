import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2

def preprocess(im):
    #Red channel input
    im = im[..., -1]
    # Crop image
    im = im[70:245, 70:345]
    immax = np.max(im)
    immin = np.min(im)
    im = 255 - im

    im = cv2.bilateralFilter(im,9,30,30)
    return im

def detect(im):

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 80 #10
    params.maxThreshold = 200


    params.filterByColor = True
    params.blobColor = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 70
    # params.maxArea = 2000

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im)


    return keypoints

if __name__ == '__main__':

    d_thresh = 50
    # im = cv2.imread("100m_200deg_ir/0710.png", cv2.IMREAD_GRAYSCALE)
    path = "100m_200deg_ir"
    save_path = "result"
    stored_kpts = []
    id_kpts = []
    mark_kpts = []
    prev_kpts = []
    prev_id_kpts = []
    for dirname, dirnames, filenames in os.walk(path):
        for frame in filenames:
            im_o = cv2.imread(os.path.join(path,frame))
            # im = cv2.imread("100m_200deg_ir/0710.png")
            # im2 = cv2.imread("100m_200deg_ir/0711.png")

            print "read image with dimentions {}, {}".format(im_o.shape[0], im_o.shape[1])

            im = preprocess(im_o)
            keypoints = detect(im)

            print "number of detections {}".format(len(keypoints))

            # mark_kpts = np.zeros_like(stored_kpts)
            if len(mark_kpts) > 0:
                mark_kpts = [m-1 for m in mark_kpts]

            for k in keypoints:
                diffs = np.zeros_like(np.array(stored_kpts))
                for i,p in enumerate(stored_kpts):
                    diffs[i] = np.linalg.norm(np.array(k.pt) - np.array(p.pt))
                if len(diffs) == 0:
                    stored_kpts.append(k)
                    K_count = 0
                    id_kpts.append(K_count)
                    mark_kpts.append(1)
                    continue

                if np.min(diffs) > d_thresh:
                    if k.pt[0]+k.size + 50 > im.shape[1]:
                        continue
                    stored_kpts.append(k)
                    K_count += 1
                    id_kpts.append(K_count)
                    mark_kpts.append(1)
                else:
                    id = np.argmin(diffs)
                    stored_kpts[id] = k
                    mark_kpts[id] += 1
                    if k.pt[0]+k.size > im.shape[1]:
                        #delete stored_kpts[id]
                        stored_kpts.pop(id)
                        id_kpts.pop(id)
                        mark_kpts.pop(id)

            remove_id = []

            # for idx,p,s in zip(range(0,len(id_kpts)),id_kpts, stored_kpts):
            #     if p in prev_id_kpts:
            #         i = prev_id_kpts.index(p)
            #         if np.sum(np.array(prev_kpts[i].pt)-np.array(s.pt)) == 0:
            #             remove_id.append(idx)
            # prev_kpts = stored_kpts
            # prev_id_kpts = id_kpts
            # remove_id = []
            for i,m in enumerate(mark_kpts):
                if m <= -1:
                    remove_id.append(i)
            for index in sorted(remove_id, reverse=True):
                del stored_kpts[index]
                del id_kpts[index]
                del mark_kpts[index]

            # prev_kpts = stored_kpts
            # prev_id_kpts = id_kpts

            stored_kpts_o = []
            for s in stored_kpts:
                s2 = cv2.KeyPoint(s.pt[0]+70, s.pt[1]+70, s.size )
                stored_kpts_o.append(s2)

            if not stored_kpts == None:
                im_with_keypoints = cv2.drawKeypoints(im_o, stored_kpts_o, np.array([]), (0, 255, 0),
                                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else:
                im_with_keypoints = im_o

            for kp, id in zip(stored_kpts_o, id_kpts):
                font = cv2.FONT_HERSHEY_SIMPLEX
                im_with_keypoints = cv2.putText(im_with_keypoints, "{}".format(id), (int(kp.pt[0]), int(kp.pt[1])), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imwrite(os.path.join(save_path,"result_{}".format(frame)),im_with_keypoints)
            # prev_keypoints = keypoints
            # cv2.imshow("frame {}".format(frame), im_with_keypoints)
            # cv2.waitKey(0)

            # plt.show()
    # im2 = preprocess(im2)


    #
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # plt.imshow(im2)
    # plt.show()
    # cv2.imshow("Image", im)
    # cv2.waitKey(0)

    #
    # im_with_keypoints = detect(im)
    # im2_with_keypoints = detect(im2)


    # fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1.imshow(im_with_keypoints)
    # # plt.show()
    # ax2.imshow(im2_with_keypoints)
    # plt.show()
    # Show blobs
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    # detector = cv2.SimpleBlobDetector()
