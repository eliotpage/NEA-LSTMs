def encode(weights):
    import numpy as np

    nums = []

    for array in weights:
        for weight in array:
            nums.append(float(weight))
    
    print(nums)


    return 'done'

if __name__ == '__main__':
    import numpy as np
    encode([np.random.uniform(-1, 1, 100), np.random.uniform(-1, 1, 100), np.random.uniform(-1, 1, 100), np.random.uniform(-1, 1, 100), np.random.uniform(-1, 1, 100)])