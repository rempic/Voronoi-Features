#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.spatial as SS
from scipy.spatial import voronoi_plot_2d
from skimage.measure import label, regionprops
from skimage.draw import ellipse, polygon, polygon_perimeter

from Voronoi_Features import MyException
from Voronoi_Features import Features
from Voronoi_Features import Utils

class Voronoi(object):
    """
    Root object from which to create the voronoi diagram

    :param  xypoints: [(x,y), ...] sets of points from which to build the voronoi diagram
    :type xypoints: List of  (x,y) pairs
    :param  image_width: width of the original image from which the points are derived
    :type image_width: int
    :param  image_width: height of the original image from which the points are derived
    :type image_width: int
    :returns: an instance of the object Voronoi
    :rtype: object
    :example:
    >>> import Voronoi_Features as VF
    >>> vor = VF.Voronoi([X,Y],1000,500)
    >>>
    >>> #VISUALIZE THE VORONOI IMAGE
    >>> fig = plt.figure(figsize=(20,15))
    >>> plt.imshow(vor.get_voronoi_map())
    """
    # to implement the class as a collection this object inherits the abstract class my_iterator
    def __init__(self, xypoints, image_width, image_height):
        try:
            self.__xypoints = xypoints
            self.__image_width = image_width
            self.__image_height = image_height
            self._VOR = None

            self.__image = np.zeros([image_height, image_width])
            self.__image_voromap = np.zeros([image_height, image_width])

            self.__make_voronoi()
            self.__make_voro_map()
        except MyException.MyException as e:
            print(e.args)


    def get_voronoi_map(self):
        """
        Returns the vornoi image in which the region pixels values are all equal to the index of the points array
        used to calculate the voronoi diagram. The voronoi image is used as data strcuture to determine the points index related to a voronoi region

        :return: voronoi image
        :rtype: 2d array
        """
        return self.__image_voromap

    def show_voronoi_plot_2d(self):
        """
        Returns the vornoi image in which the region pixels values are all equal to the index of the points array
        used to calculate the voronoi diagram. The voronoi image is used as data strcuture to determine the points index related to a voronoi region

        :return: voronoi image
        :rtype: 2d array
        """
        if self._VOR == None:
            print("[Error] It is not possible to show the _VOR is Null")
            return -1
        
        voronoi_plot_2d(self._VOR)
        
        return 0
    
    def __make_voronoi(self):
        """
        Makes the vornoi image with region pixels values equal to the index for the points array used to build the
        voronoi. In this class the voro-image is used as data strcuture to determine the points index refering to
        the voro-regions
        """
        self._VOR = SS.Voronoi(self.__xypoints)


    def __make_voro_map(self):
        """
        Makes the vornoi image with region pixels values equal to the index for the points array used to build the
        voronoi. In this class the voro-image is used as data strcuture to determine the points index refering to
        the voro-regions
        """
        n = len(self.__xypoints)

        # loop on all points (centroids) of the single voronoi region
        for xy_idx in range(0,n):
            # gets the index of a voronoi region
            reg_idx = self._VOR.point_region[xy_idx]
            # gets the voronoi region
            region = self._VOR.regions[reg_idx]
            # calculates the polygon of the region
            poly = self._VOR.vertices[region]
            #print("polygon:{}".format(poly))
            #print("prod:{}".format(poly.min()))
            c0 = poly.min()
            c1 = poly[:,0].max()
            c2 = poly[:,1].max()

            if (c0>0) & (c2<self.__image_width) & (c1<self.__image_height):
                #  build the x and y vectors of the x and y polygon coordinates
                x = poly[:,0]
                y = poly[:,1]
                # fills with values 1 the pixels of the region polygon on the temporary refering to the entire voronoi diagram
                rr, cc = polygon(x, y)
                rrp,ccp = polygon_perimeter(x,y)
                #print("rr:{}".format(rr.max()))
                #print("cc:{}".format(cc.max()))
                self.__image_voromap[rr, cc] = xy_idx
                self.__image_voromap[rrp, ccp] = 0

        return self.__image_voromap


    def features(self, props_name):
        """
        Measure the values of the specified  property/measure name (e.g., 'area') for all voronoi regions.

        :param prop_name: name of the property to measure (e.g, 'area')
        :type prop_name: string
        :returns: Feature object
        :rtype: Object

        :example:
        >>> import Voronoi_Features as VF
        >>> voro = VF.Images(folder_name)
        >>> features = voro.features(['area','perimeter'])
        >>> features.get_data_frame().head()

        The following properties can be accessed as attributes or keys:

        **area** : int
            Number of pixels of region.
        **bbox** : tuple
            Bounding box ``(min_row, min_col, max_row, max_col)``.
            Pixels belonging to the bounding box are in the half-open interval
            ``[min_row; max_row)`` and ``[min_col; max_col)``.
        **bbox_area** : int
            Number of pixels of bounding box.
        **centroid** : array
            Centroid coordinate tuple ``(row, col)``.
        **convex_area** : int
            Number of pixels of convex hull image.
        **convex_image** : (H, J) ndarray
            Binary convex hull image which has the same size as bounding box.
        **coords** : (N, 2) ndarray
            Coordinate list ``(row, col)`` of the region.
        **eccentricity** : float
            Eccentricity of the ellipse that has the same second-moments as the
            region. The eccentricity is the ratio of the focal distance
            (distance between focal points) over the major axis length.
            The value is in the interval [0, 1).
            When it is 0, the ellipse becomes a circle.
        **equivalent_diameter** : float
            The diameter of a circle with the same area as the region.
        **euler_number** : int
            Euler characteristic of region. Computed as number of objects (= 1)
            subtracted by number of holes (8-connectivity).
        **extent** : float
            Ratio of pixels in the region to pixels in the total bounding box.
            Computed as ``area / (rows * cols)``
        **filled_area** : int
            Number of pixels of filled region.
        **filled_image** : (H, J) ndarray
            Binary region image with filled holes which has the same size as
            bounding box.
        **image** : (H, J) ndarray
            Sliced binary region image which has the same size as bounding box.
        **inertia_tensor** : (2, 2) ndarray
            Inertia tensor of the region for the rotation around its mass.
        **inertia_tensor_eigvals** : tuple
            The two eigen values of the inertia tensor in decreasing order.
        **intensity_image** : ndarray
            Image inside region bounding box.
        **label** : int
            The label in the labeled input image.
        **local_centroid** : array
            Centroid coordinate tuple ``(row, col)``, relative to region bounding
            box.
        **major_axis_length** : float
            The length of the major axis of the ellipse that has the same
            normalized second central moments as the region.
        **max_intensity** : float
            Value with the greatest intensity in the region.
        **mean_intensity** : float
            Value with the mean intensity in the region.
        **min_intensity** : float
            Value with the least intensity in the region.
        **minor_axis_length** : float
            The length of the minor axis of the ellipse that has the same
            normalized second central moments as the region.
        **moments** : (3, 3) ndarray
            Spatial moments up to 3rd order::
                m_ji = sum{ array(x, y) * x^j * y^i }
            where the sum is over the `x`, `y` coordinates of the region.
        **moments_central** : (3, 3) ndarray
            Central moments (translation invariant) up to 3rd order::
                mu_ji = sum{ array(x, y) * (x - x_c)^j * (y - y_c)^i }
            where the sum is over the `x`, `y` coordinates of the region,
            and `x_c` and `y_c` are the coordinates of the region's centroid.
        **moments_hu** : tuple
            Hu moments (translation, scale and rotation invariant).
        **moments_normalized** : (3, 3) ndarray
            Normalized moments (translation and scale invariant) up to 3rd order::
                nu_ji = mu_ji / m_00^[(i+j)/2 + 1]
            where `m_00` is the zeroth spatial moment.
        **orientation** : float
            Angle between the X-axis and the major axis of the ellipse that has
            the same second-moments as the region. Ranging from `-pi/2` to
            `pi/2` in counter-clockwise direction.
        **perimeter** : float
            Perimeter of object which approximates the contour as a line
            through the centers of border pixels using a 4-connectivity.
        **solidity** : float
            Ratio of pixels in the region to pixels of the convex hull image.
        **weighted_centroid** : array
            Centroid coordinate tuple ``(row, col)`` weighted with intensity
            image.
        **weighted_local_centroid** : array
            Centroid coordinate tuple ``(row, col)``, relative to region bounding
            box, weighted with intensity image.
        **weighted_moments** : (3, 3) ndarray
            Spatial moments of intensity image up to 3rd order::
                wm_ji = sum{ array(x, y) * x^j * y^i }
            where the sum is over the `x`, `y` coordinates of the region.
        **weighted_moments_central** : (3, 3) ndarray
            Central moments (translation invariant) of intensity image up to
            3rd order::
                wmu_ji = sum{ array(x, y) * (x - x_c)^j * (y - y_c)^i }
            where the sum is over the `x`, `y` coordinates of the region,
            and `x_c` and `y_c` are the coordinates of the region's weighted
            centroid.
        **weighted_moments_hu** : tuple
            Hu moments (translation, scale and rotation invariant) of intensity
            image.
        **weighted_moments_normalized** : (3, 3) ndarray
            Normalized moments (translation and scale invariant) of intensity
            image up to 3rd order::
                wnu_ji = wmu_ji / wm_00^[(i+j)/2 + 1]
            where ``wm_00`` is the zeroth spatial moment (intensity-weighted area).


        .. [1] http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
        """
        try:
            df = pd.DataFrame()

            labs = label(self.__image_voromap)
            region_props = regionprops(labs)
            m = len(props_name)
            num_reg = len(region_props)

            # add the ID by using the __image_voromap
            IDs = []
            for preg in region_props:
                centroid = getattr(preg, 'centroid')
                ID = self.__image_voromap[np.int(centroid[0]), np.int(centroid[1])]
                IDs.append(np.int(ID))
            Utils.insert_values('id',df,IDs)

            # add the prop values
            for pname in props_name:
                vals = []
                for preg in region_props:
                    vals.append(getattr(preg, pname))
                Utils.insert_values(pname,df,vals)

            return Features.Features(df)
        except MyException.MyException as e:
            print(e.args)
            return None
