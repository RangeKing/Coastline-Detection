# -*- coding: utf-8 -*-
'''
@Author: Range King
Start Building on 2019-8-16
'''
# Import the libs needed
from __future__ import division
import os
import sys
import cv2
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo.gdalconst import *

if __name__ == '__main__':
    ''' Write Chinese'''
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # gdal.SetConfigOption("SHAPE_ENCODING", "GBK")

    # ------
    # Read the data
    # ------
    # Detect the Lib needed
    gdal_version_num = int(gdal.VersionInfo('VERSION_NUM'))
    if gdal_version_num < 1100000:
        sys.exit('ERROR: Python bindings of GDAL 1.10 or later required')
    if cv2.__version__ < '3.2.0':
        sys.exit('ERROR: Python bindings of OpenCV 3.1.0 or later required')

    print 'gdal_version_num = ', gdal_version_num, 'cv_version_num = ', cv2.__version__

    # Read GeoTiff by GDAL
    FileName = input('input path and filename(Example: r\'D:\\data\inData.tif\')')
    # FileName = r'F:\Research\Dr\Coastal Line\data\YongXingDao\SV1-02_20170511_L1B0000490424_SV2018060432175-MUX.tif'
    # FileName = r'F:\Research\Dr\Coastal Line\data\ZhaiRuoShan\GF2_PMS1_E122.1_N29.9_20180115_L1A0002933472-MSS1.tif'
    img = gdal.Open(FileName, GA_ReadOnly)
    if img is None:
        sys.exit('ERROR: Could not open ' + FileName)
    cols = img.RasterXSize
    rows = img.RasterYSize
    bands = img.RasterCount
    pixels = cols * rows
    print img.GetDriver().ShortName

    # Transform data read by GDAL to the format that OpenCV can read
    band_names = locals()
    for i in xrange(1, bands + 1):
        band_names['band%s' % i] = img.GetRasterBand(i).ReadAsArray(0, 0, cols, rows)

    # Normalization
    band_val_max = np.max([np.max(band1), np.max(band2), np.max(band3), np.max(band4)])
    band1_nor = band1 / band_val_max
    band2_nor = band2 / band_val_max
    band3_nor = band3 / band_val_max
    band4_nor = band4 / band_val_max

    # Calculate mean and std
    band1_nor_mean = np.mean(band1_nor)
    band2_nor_mean = np.mean(band2_nor)
    band3_nor_mean = np.mean(band3_nor)
    band4_nor_mean = np.mean(band4_nor)
    band1_nor_std = np.std(band1_nor)
    band2_nor_std = np.std(band2_nor)
    band3_nor_std = np.std(band3_nor)
    band4_nor_std = np.std(band4_nor)

    # ------
    # Mask
    # ------
    # Land detection
    island_coef = 1.0
    land_coef = 2.0
    band1_nor_masked = band1_nor
    band1_nor_masked[band1_nor > (band1_nor_mean + island_coef * band1_nor_std)] = 1
    band1_nor_masked[band1_nor > (band1_nor_mean + land_coef * band1_nor_std)] = 2
    band2_nor_masked = band2_nor
    band2_nor_masked[band2_nor > (band2_nor_mean + island_coef * band2_nor_std)] = 1
    band2_nor_masked[band2_nor > (band2_nor_mean + land_coef * band2_nor_std)] = 2
    band3_nor_masked = band3_nor
    band3_nor_masked[band3_nor > (band3_nor_mean + island_coef * band3_nor_std)] = 1
    band3_nor_masked[band3_nor > (band3_nor_mean + land_coef * band3_nor_std)] = 2
    band4_nor_masked = band4_nor
    band4_nor_masked[band4_nor > (band4_nor_mean + island_coef * band4_nor_std)] = 1
    band4_nor_masked[band4_nor > (band4_nor_mean + land_coef * band4_nor_std)] = 2
    land = np.zeros([rows, cols], dtype=float)
    island = np.zeros([rows, cols], dtype=float)
    land[
        (band1_nor_masked > 1) & (band2_nor_masked > 1) & (band3_nor_masked > 1) & (band4_nor_masked > 1)] = 1
    island[
        (band1_nor_masked > .4) & (band2_nor_masked > .4) & (band3_nor_masked > .4) & (band4_nor_masked > .4)] = 1
    # Read the necessary variables
    geotransform = img.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]

    # band2[band2 == 0] = 1
    # band4[band4 == 0] = 1
    ndwi = (band2_nor - band4_nor) / (band2_nor + band4_nor)
    ndvi = (band4_nor - band3_nor) / (band4_nor + band3_nor)

    # ------
    # Canny Edge Detection
    # ------
    ndwi_Gau = cv2.GaussianBlur(ndwi, (3, 3), 0)
    ndvi_Gau = cv2.GaussianBlur(ndvi, (3, 3), 0)
    # band1_8u = (band1_masked > 0) * (band1_masked - band1.min() + 1) * 255 / (band1.max() - band1.min())
    # band1_8u = band1_masked / 256

    th1, ndwi_binary = cv2.threshold(ndwi_Gau, 0, 1, cv2.THRESH_BINARY_INV)
    th2, ndvi_binary = cv2.threshold(ndvi_Gau, 0.5, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ndwi_closing = cv2.morphologyEx(ndwi_binary, cv2.MORPH_CLOSE, kernel, iterations=10)
    classification = np.zeros([rows, cols], dtype=float)
    classification[land != 0] = 1
    classification[(land == 0) & (ndvi_binary == 1)] = 2
    classification[(classification != 1) & (classification != 2) & (ndwi_binary == 1)] = 3

    # ---
    # Output1
    # ---
    # Create the output Layer
    outShapefile = input('output path and island classificasion filename(Example: r\'D:\\result\outData.shp\')')
    # outShapefile = r"F:\Research\Dr\Coastal Line\Python\shp_polygon_test5_YXD.shp"
    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)
    # get the spatial reference from input image
    srs = osr.SpatialReference()
    srs.ImportFromWkt(img.GetProjectionRef())

    # create layer with proj
    outDataSource = outDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer(outShapefile, srs, geom_type=ogr.wkbPolygon)

    # Add class column (1,2...) to shapefile
    # ds = gdal_array.OpenArray(binary)
    driver = img.GetDriver()
    # print driver

    tempFileName = 'Coastline_temp1.tif'
    outDs = driver.Create(tempFileName, cols, rows, 1, GDT_Float32)
    # outDs = driver.Create(r"F:\Research\Dr\Coastal Line\Python\shp_polygon_test_YXD.tif", cols, rows, 1, GDT_Int32)
    # if outDs is None:
    #     print 'Could not create temp file'
    #     sys.exit(1)

    # write the data
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(classification, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)

    # georeference the image and set the projection
    outDs.SetGeoTransform(img.GetGeoTransform())
    outDs.SetProjection(img.GetProjection())

    # Add the fields we're interested in
    outLayer.CreateField(ogr.FieldDefn("ClassID", ogr.OFTReal))
    field_LULC = ogr.FieldDefn("LULC", ogr.OFTString)
    field_LULC.SetWidth(24)
    outLayer.CreateField(field_LULC)

    outLayer.CreateField(ogr.FieldDefn("Area", ogr.OFTReal))
    # outLayer.CreateField(ogr.FieldDefn("Length", ogr.OFTReal))
    outPoly = gdal.Polygonize(outBand, None, outLayer, 0, [], callback=None)

    # Process the text file and add the attributes and features to the shapefile
    for feature in outLayer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        length = geom.Boundary().Length()
        if feature["ClassID"] == 0:
            feature.SetField("LULC", "unclassified")
        elif feature["ClassID"] == 1:
            feature.SetField("LULC", "land")
        elif feature["ClassID"] == 2:
            feature.SetField("LULC", "vegetation")
        elif feature["ClassID"] == 3:
            feature.SetField("LULC", "building")

        # Set the attributes using the values
        feature.SetField("Area", area)
        # feature.SetField("Length", length)
        outLayer.SetFeature(feature)
    outDataSource.Destroy()

    # ---
    # Output1
    # ---
    # Create the output Layer
    outShapefile = input('output path and island coastline filename(Example: r\'D:\\result\outData.shp\')')
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)
    # get the spatial reference from input image
    srs = osr.SpatialReference()
    srs.ImportFromWkt(img.GetProjectionRef())

    # create layer with proj
    outDataSource = outDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer(outShapefile, srs, geom_type=ogr.wkbPolygon)

    # Add class column (1,2...) to shapefile
    # ds = gdal_array.OpenArray(binary)
    driver = img.GetDriver()
    # print driver

    tempFileName = 'Coastline_temp2.tif'
    outDs = driver.Create(tempFileName, cols, rows, 1, GDT_Float32)
    # outDs = driver.Create(r"F:\Research\Dr\Coastal Line\Python\shp_polygon_test_YXD.tif", cols, rows, 1, GDT_Int32)
    # if outDs is None:
    #     print 'Could not create temp file'
    #     sys.exit(1)

    # write the data
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(ndwi_closing, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)

    # georeference the image and set the projection
    outDs.SetGeoTransform(img.GetGeoTransform())
    outDs.SetProjection(img.GetProjection())

    # Add the fields we're interested in
    outLayer.CreateField(ogr.FieldDefn("ClassID", ogr.OFTReal))
    field_LULC = ogr.FieldDefn("LULC", ogr.OFTString)
    field_LULC.SetWidth(24)
    outLayer.CreateField(field_LULC)

    outLayer.CreateField(ogr.FieldDefn("Area", ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn("Length", ogr.OFTReal))
    # outLayer.CreateField(ogr.FieldDefn("Length", ogr.OFTReal))
    outPoly = gdal.Polygonize(outBand, None, outLayer, 0, [], callback=None)
    for feature in outLayer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        length = geom.Boundary().Length()
        if feature["ClassID"] == 1:
            feature.SetField("LULC", "island")
        # Set the attributes using the values
        feature.SetField("Area", area)
        feature.SetField("Length", length)
        outLayer.SetFeature(feature)
    outDataSource.Destroy()
    print('Done')
