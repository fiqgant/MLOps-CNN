import numpy as np
from keras.preprocessing import image


test_image = image.load_img('./Data/test/angry/38.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)


training_data.class_indices
if result[0][0] == 1:
    prediction = '11'
elif result[0][1] == 2:
    prediction = '22'
elif result[0][2] == 3:
    prediction = '33'
elif result[0][3] == 4:
    prediction = '44'
elif result[0][4] == 5:
    prediction = '55'
elif result[0][5] == 6:
    prediction = '66'
else:
    prediction = '77'


print(result)
max_value = max(result)
max_index_col = np.argmax(result, axis=1)
max_index_col