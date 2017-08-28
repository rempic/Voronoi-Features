.. Voronoi features extraction documentation master file, created by
   sphinx-quickstart on Mon Aug 14 17:04:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Voronoi Features Extraction
============================

This package allows the fast extraction of Voroni features from a set of images and includes 38 type of measurements for each single Voronoi tassel.
The resulting data frame can be used as training and testing set for machine learning classifier

The package can be used for a variety of applications. It was originally used to measure single cell nuclei features
from microscopy images (see figure below), and for extracting information from maps for city density analysis and modeling

.. image:: _static/1.png
   :width: 600px
   :alt: alternate text

Visualizations of Voronoi diagrams:

.. image:: _static/3.png
  :width: 600px
  :alt: alternate text

.. image:: _static/4.png
 :width: 600px
 :alt: alternate text

An example of a data set returned from the Vononoi features extraction. The package allows the measurement
of more then 30 type of properties for a single Voronoi region

 .. image:: _static/5.png
  :width: 600px
  :alt: alternate text

See classes/modules documentation for more details about the package code

Working in progress ...

Contents:
============

.. toctree::
   :maxdepth: 2

   tutorial.rst
   code.rst
