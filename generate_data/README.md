#Generating data

### `tools` is a directory that point to https://github.com/affromero/FasterRCNN-Faces 
`git clone --recursive https://github.com/affromero/FasterRCNN-Faces`
In order to keep lightweight this repo, I did not use the corresponding submodule.

-----

### OF
Inside `OF` folder there are the scripts to generate the Optical Flow is calculated based on the BP4D folder directory. All you need to do is to place the root and target destination of the OF in `OF/OF_BP4D.py` and run it. 

There is one missing file `broxDir` inside OF folder. In order to download it, please follow this [link](https://drive.google.com/open?id=1x9O1UgDjAwH_Hk6qUntDb5C8SV-3HB6d). 
