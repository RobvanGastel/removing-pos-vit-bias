# Post-Training to Remove Positional Bias
Using Franca's RASA (Venkataramanan et al., 2025) to remove positional bias in of other SSL pretrained ViTs is simple and provides an increase in this downstream performance. I evaluated the performance with OverClustering (Ziegler & Asano, 2022) and obtain a 1-3% performance boost on the validation set for different ViT model sizes. Since the original codebase for RASA is not easy to reuse I adjusted it to easily put in any pre-trained encoder. 

I have observed a increase in DINOv2, DINOv3 so far.

![](/assets/market_cosine_sim.png?raw=true)


Things I still want to try and improve:
- Measure linear segmentation performance with RASA post-training.
- Evaluate performance different datasets for downstream and post-training.
- Can post-training with gram anchorring of DINOv3s distilled models also improve scaling to higher resolution images.

## Results

**Pascal VOC** 
Performance on the Pascal VOC2012 validation set with ViT-S for OverClustering (Ziegler & Asano, 2022).
<table style="margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="padding:6px 10px;">k</th>
      <th style="padding:6px 10px;">Validation mIoU</th>
      <th style="padding:6px 10px;">RASA Validation mIoU</th>
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
      <td align="right" style="padding:6px 10px;">—</td>
      <td align="right" style="padding:6px 10px;">—</td>
    </tr>
  </tbody>
</table>

Performance on the Pascal VOC2012 validation set with ViT-B for OverClustering (Ziegler & Asano, 2022).

<table style="margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="padding:6px 10px;">k</th>
      <th style="padding:6px 10px;">Validation mIoU</th>
      <th style="padding:6px 10px;">RASA Validation mIoU</th>
      <th style="padding:6px 10px;">Δ vs Original</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right" style="padding:6px 10px;">21</td>
      <td align="right" style="padding:6px 10px;">-</td>
      <td align="right" style="padding:6px 10px;">-</td>
      <td align="right" style="padding:6px 10px;">-</td>
    </tr>
    <tr>
      <td align="right" style="padding:6px 10px;">100</td>
      <td align="right" style="padding:6px 10px;">-</td>
      <td align="right" style="padding:6px 10px;">-</td>
      <td align="right" style="padding:6px 10px;">-</td>
    </tr>
    <tr>
      <td align="right" style="padding:6px 10px;">300</td>
      <td align="right" style="padding:6px 10px;">-</td>
      <td align="right" style="padding:6px 10px;">—</td>
      <td align="right" style="padding:6px 10px;">—</td>
    </tr>
  </tbody>
</table>

## References
Venkataramanan, S., Pariza, V., Salehi, M., Knobel, L., Gidaris, S., Ramzi, E., Bursuc, A., & Asano, Y. M. (2025). Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning (No. arXiv:2507.14137). arXiv. https://doi.org/10.48550/arXiv.2507.14137

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., … Bojanowski, P. (2025). DINOv3 (No. arXiv:2508.10104). arXiv. https://doi.org/10.48550/arXiv.2508.10104

Ziegler, A., & Asano, Y. M. (2022). Self-Supervised Learning of Object Parts for Semantic Segmentation (No. arXiv:2204.13101). arXiv. https://doi.org/10.48550/arXiv.2204.13101
