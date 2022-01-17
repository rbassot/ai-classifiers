### CIFAR Dataset:
- 10 labels, each with a label_name; labels numbered 0-9 representing label_list index

data_dict['data']:
    - each one of 10,000 rows: 3072 entries of uint8 type
    - each one of these rows **represents a 32x32 colour image (1024 pixels)**
    In each row:
        - first 1024 entries are the red channel
        - second 1024 = green channel
        - third 1024 = blue channel
        - image is stored in row-major order (ie. 33rd entry == 1st entry of the 2nd row)