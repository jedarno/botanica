
def generate_class_weights(images, nclasses):
    """
    Weight finding function adapted from example given by Prajot Kuvalekar
    https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample

    Parameters
    ----------
    images : list
        list of image class pairs from torch iterator
    nclasses : int
        Total number of classes in data set
    """
    n_images = len(images)
    count_per_class = [0] * nclasses

    for _, image_class in images:
        count_per_class[image_class] += 1

    weight_per_class = [0.] * nclasses

    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])

    weights = [0] * n_images

    for idx, (image, image_class) in enumerate(images):
        weights[idx] = weight_per_class[image_class]

    return weights


