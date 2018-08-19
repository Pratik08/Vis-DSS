/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#include "ImageUtils.h"
// Compute the average color histogram for frames between startframe and endframe
cv::Mat GetSquareImage( const cv::Mat& img, int target_width)
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}

void tile(std::vector<cv::Mat> &src, cv::Mat &dst, int grid_x, int grid_y, int budget) {
    // patch size
    int width  = dst.cols/grid_x;
    int height = dst.rows/grid_y;
    // iterate through grid
    int k = 0;
    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {
            if (k < budget)
            {
                cv::Mat s = src[k];
                cv::resize(s,s,cv::Size(width,height));
                s.copyTo(dst(cv::Rect(j*width,i*height,width,height)));
            }
            k++;
        }
    }
}
