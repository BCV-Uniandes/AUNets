# AUNets
This repo provides a PyTorch implementation for AUNets. AUNets relies on the power of having independent and binaries CNNs for each facial expression. It works with the hollistical facial image *i.e.*, no keypoints or facial aligment needed. 

Paper: https://arxiv.org/pdf/1704.07863.pdf

## VGG16 Architecture
![](misc/arqs.png)

## Network (VGG - Fig (a))
![](misc/VGG16-OF_None.png)

Other Optical Flow stream architectures are found in `misc` folder

## Requirements

- Python 2.7 
- PyTorch 0.3.1
- Tensorflow (only if tensorboard flag)
- Other modules in `requirements.txt`

## Weights (*.pth models*)

As each model can be as heavy as 1GB for a total of 36GB per network variant (12 AUs - 3 folds). Please feel free to create an issue asking for the weights, I will supply a temporal link to download them. I am doing this because I cannot keep open a link with at least ~40GB forever.  

## Results using `misc/VGG16-OF_Horizontal.png`)

<table>
<thead>
<tr class="header">
<th style="text-align: right;"><strong>AU</strong></th>
<th style="text-align: center;">1</th>
<th style="text-align: center;">2</th>
<th style="text-align: center;">4</th>
<th style="text-align: center;">6</th>
<th style="text-align: center;">7</th>
<th style="text-align: center;">10</th>
<th style="text-align: center;">12</th>
<th style="text-align: center;">14</th>
<th style="text-align: center;">15</th>
<th style="text-align: center;">17</th>
<th style="text-align: center;">23</th>
<th style="text-align: center;">24</th>
<th style="text-align: center;"><strong><em>Av.</em></strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;"><p><strong>F1</strong></p></td>
<td style="text-align: center;">53.4</td>
<td style="text-align: center;">44.7</td>
<td style="text-align: center;">55.8</td>
<td style="text-align: center;">79.2</td>
<td style="text-align: center;">78.1</td>
<td style="text-align: center;">83.1</td>
<td style="text-align: center;">88.4</td>
<td style="text-align: center;">66.6</td>
<td style="text-align: center;">47.5</td>
<td style="text-align: center;">62.0</td>
<td style="text-align: center;">47.3</td>
<td style="text-align: center;">49.7</td>
<td style="text-align: center;"><strong>63.0</strong></td>
</tr>
</tbody>
</table>
