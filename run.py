from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2

path_to_networks = '20170512-110547/'
path_to_images = 'test_images/'
graph_filename = 'model-20170512-110547.ckpt-250000.data-00000-of-00001'
image_filename = path_to_images + 'people4.jpg'
id_folder = 'history_curated'

def show_cam(image, matching_id, bb):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image, matching_id, (bb[0], bb[3]), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

#Load categories
categories = []
with open('names.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        categories.append(cat)
    f.close()
    print('Number of categories:', len(categories))

graph = device.AllocateGraph(graphfile)

image = misc.imread(image_filename)

print('Start download to NCS...')
graph.LoadTensor(frame, 'user object')

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

id_dataset = id_data.get_id_data(
    id_folder, pnet, rnet, onet, sess, embeddings, images_placeholder, phase_train_placeholder)

# Load embeddings
face_patches, _, _ = detect_and_align.align_image(
    image, pnet, rnet, onet)
aligned_images = aligned_images + face_patches
aligned_image_paths = aligned_image_paths + \
    [image_paths[i]] * len(face_patches)

aligned_images = np.stack(aligned_images)

feed_dict = {images_placeholder: aligned_images,
             phase_train_placeholder: False}
embs = sess.run(embeddings, feed_dict=feed_dict)

for i in range(len(embs)):
    misc.imsave('output/outfile' + str(i) + '.jpg', aligned_images[i])
    matching_id, dist, _ = find_matching_id(id_dataset, embs[i, :])
    if matching_id:
        print('Found match %s for %s! Distance: %1.4f File: outfile%s'  %
              (matching_id, aligned_image_paths[i], dist, i))
    else:
        print('Couldn\'t find match for %s' % (aligned_image_paths[i]))

face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(
    frame, pnet, rnet, onet)

if len(face_patches) > 0:
    face_patches = np.stack(face_patches)
    feed_dict = {images_placeholder: face_patches,
                 phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)
    for i in range(len(embs)):
        matching_id, dist, accuracy = find_matching_id(
        id_dataset, embs[i, :])
        bb = padded_bounding_boxes[i]
        if matching_id:
            print('Hi %s! Distance: %1.4f' %
                  (matching_id, dist))
        else:
            matching_id = 'You are special!'
            print('Unknown! Couldn\'t find match.')

output, userobj = graph.GetResult()

top_inds = output.argsort()[::-1][:5]

print(''.join(['*' for i in range(79)]))
print('inception-v3 on NCS')
print(''.join(['*' for i in range(79)]))
for i in range(5):
    print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

print(''.join(['*' for i in range(79)]))
graph.DeallocateGraph()
device.CloseDevice()
print('Finished')
