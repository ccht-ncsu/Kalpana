<img src="adds/imgs/kalpana.PNG" width="1028"/>

Kalpana is a Python module to convert *ADCIRC* output files to geospatial vector formats (shapefile or kmz), and to downscale the maximum water elevations onto a higher-resolution raster. 

## *ADCIRC* to geospatial vector formats

Kalpana can convert time-varying outputs (e.g. *fort.63.nc*, *swan_HS.63.nc*) and time-constant outputs (*maxele.63.nc*) into polylines or polygons in both shapefile and kmz formats. 
Kalpana was developed originally by Rosemary Cyriac, and her efforts were aided by the work of Rich Signell and Rusty Holleman to generate shapefiles from *ADCIRC* results. Then, Jason Fleming improved Kalpana and incorporated it into the ADCIRC Surge Guidance System (ASGS).

## Downscaling

Kalpana can downscale the maximum water elevations (*maxele.63.nc*) to a higher-resolution raster by considering small-scale topographic features. This process can provide a more-accurate representation of the inundation extent. 
As part of the downscaling, the water surface can be expanded outward to intersect with the ground surface, beyond the extent predicted by ADCIRC. This expansion can be done in two ways: the static method was developed by Nelson Tull, and then the head-loss method was developed by Carter Rucker. The details can be found in [this paper](https://link.springer.com/epdf/10.1007/s11069-021-04634-8?sharing_token=5GBxenc0qDVGHm3BGk6KhPe4RwlQNchNByi7wbcMAY69maaLpgXTBxca-OorPGWBn2w2ySSkXhIRhNeWoyNx8-ituX0UqAcNj_LDMh_kFz6sCpb5e882TbeHKiKpzRd_j4XfVH_6ONriheKYxx2CECQI07z23OD-pFrCALWfyVc=). The schematics below show the downscaling process.

**Storm surge expansion**

<img src="adds/imgs/kalpana_extend.png" width="512"/>

**Storm surge contraction**

<img src="adds/imgs/kalpana_shrink.png" width="512"/>

## Updated version

Kalpana was updated to python 3 and upgraded by Tomás Cuevas as a part of his MSc research. 
Instructions for using Kalpana can be found in the examples folder. 
For any question, comment or suggestion please send an email to tomascuevas@gmail.com or open an *Issue*.
