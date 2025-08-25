# Post-Training to Remove Positional Bias
Using Franca's "Removal of Absolute Spatial Attributes" Post-Training (RASA) (Venkataramanan et al., 2025) to remove positional bias in of other SSL pretrained ViTs is simple and provides an increase in this downstream performance. I evaluated the performance with OverClustering (Ziegler & Asano, 2022) and obtain a 1-3% performance boost on the validation set for different ViT model sizes. Since the original codebase for RASA is not easy to reuse I adjusted it to easily put in any pre-trained encoder. I have observed an increase in performance for the DINOv2, DINOv3 encoders.


![](/assets/market_cosine_sim.png?raw=true)

<p>
    <a href= "https://colab.research.google.com/github/RobvanGastel/removing-pos-vit-bias/blob/main/visualization.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

This image displays patch cosine similarity between a selected patch token and the other patches, like on page 4 of the DINOv3 paper (Siméoni et al., 2025). This quantative evaluation helps us see how well it can distinguish between object types in the image. In the `visualization.ipynb` I evaluate what encoder size and RASA post-training does to the performance of the model. Smaller models still struggle to produce good cosine similarities. See the `visualization.ipynb` notebook or test it for yourself in Google collab.

Things I am still evaluating:
- Measure linear segmentation performance with RASA post-training.
- Evaluate performance different datasets for downstream and post-training.
- Can post-training with gram anchorring of DINOv3s distilled models also improve scaling to higher resolution images.

## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name dino python=3.11
conda activate dino
# Run the code, adjust the ./configs/rasa.yml or argparse flags
python main.py --exp_name "rasa_vits"
```

## Results

**Pascal VOC2012** \
Performance on the validation set with the DINOv3 ViT-S encoder for OverClustering (Ziegler & Asano, 2022) with k={21, 100, 300} on 40/90 batches of the validation set with batch size 16 (due to compute constraints).
<table style="margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="padding:6px 10px;">k</th>
      <th style="padding:6px 10px;">Validation mIoU</th>
      <th style="padding:6px 10px;">with RASA Validation mIoU</th>
      <th style="padding:6px 10px;">Δ vs Original</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right" style="padding:6px 10px;">21</td>
      <td align="right" style="padding:6px 10px;">15.67%</td>
      <td align="right" style="padding:6px 10px;">16.12%</td>
      <td align="right" style="padding:6px 10px;">+0.45%</td>
    </tr>
    <tr>
      <td align="right" style="padding:6px 10px;">100</td>
      <td align="right" style="padding:6px 10px;">46.56%</td>
      <td align="right" style="padding:6px 10px;">47.64%</td>
      <td align="right" style="padding:6px 10px;">+1.08%</td>
    </tr>
    <tr>
      <td align="right" style="padding:6px 10px;">300</td>
      <td align="right" style="padding:6px 10px;">59.94%</td>
      <td align="right" style="padding:6px 10px;">59.14%</td>
      <td align="right" style="padding:6px 10px;">+0.80%</td>
    </tr>
  </tbody>
</table>

Performance on the validation set with the DINOv3 ViT-B encoder for OverClustering (Ziegler & Asano, 2022) with k={21, 100, 300} on 40/90 batches of the validation set with batch size 16 (due to compute constraints).

<table style="margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="padding:6px 10px;">k</th>
      <th style="padding:6px 10px;">Validation mIoU</th>
      <th style="padding:6px 10px;">with RASA Validation mIoU</th>
      <th style="padding:6px 10px;">Δ vs Original</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right" style="padding:6px 10px;">21</td>
      <td align="right" style="padding:6px 10px;">20.07%</td>
      <td align="right" style="padding:6px 10px;">21.56%</td>
      <td align="right" style="padding:6px 10px;">+1.56%</td>
    </tr>
    <tr>
      <td align="right" style="padding:6px 10px;">100</td>
      <td align="right" style="padding:6px 10px;">54.22%</td>
      <td align="right" style="padding:6px 10px;">51.30%</td>
      <td align="right" style="padding:6px 10px;">+2.92%</td>
    </tr>
    <tr>
      <td align="right" style="padding:6px 10px;">300</td>
      <td align="right" style="padding:6px 10px;">68.03%</td>
      <td align="right" style="padding:6px 10px;">66.60%</td>
      <td align="right" style="padding:6px 10px;">-1.43%</td>
    </tr>
  </tbody>
</table>

I picked the best weights based on the an intermediate evaluation with $k=21$. Therefore, it might work suboptimal for larger $k$. 

## References
Venkataramanan, S., Pariza, V., Salehi, M., Knobel, L., Gidaris, S., Ramzi, E., Bursuc, A., & Asano, Y. M. (2025). Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning (No. arXiv:2507.14137). arXiv. https://doi.org/10.48550/arXiv.2507.14137

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., … Bojanowski, P. (2025). DINOv3 (No. arXiv:2508.10104). arXiv. https://doi.org/10.48550/arXiv.2508.10104

Ziegler, A., & Asano, Y. M. (2022). Self-Supervised Learning of Object Parts for Semantic Segmentation (No. arXiv:2204.13101). arXiv. https://doi.org/10.48550/arXiv.2204.13101
