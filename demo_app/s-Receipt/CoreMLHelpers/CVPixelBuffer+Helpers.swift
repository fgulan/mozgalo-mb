/*
  Copyright (c) 2017 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import Foundation
import Accelerate
import CoreImage

/**
 Creates a RGB pixel buffer of the specified width and height.
*/
public func createPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
  var pixelBuffer: CVPixelBuffer?
  let status = CVPixelBufferCreate(nil, width, height,
                                   kCVPixelFormatType_32BGRA, nil,
                                   &pixelBuffer)
  if status != kCVReturnSuccess {
    print("Error: could not create resized pixel buffer", status)
    return nil
  }
  return pixelBuffer
}

/**
 First crops the pixel buffer, then resizes it.
*/
public func resizePixelBuffer(_ srcPixelBuffer: CVPixelBuffer,
                              cropX: Int,
                              cropY: Int,
                              cropWidth: Int,
                              cropHeight: Int,
                              scaleWidth: Int,
                              scaleHeight: Int) -> CVPixelBuffer? {

  CVPixelBufferLockBaseAddress(srcPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
  guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
    print("Error: could not get pixel buffer base address")
    return nil
  }
  let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
  let offset = cropY*srcBytesPerRow + cropX*4
  var srcBuffer = vImage_Buffer(data: srcData.advanced(by: offset),
                                height: vImagePixelCount(cropHeight),
                                width: vImagePixelCount(cropWidth),
                                rowBytes: srcBytesPerRow)

  let destBytesPerRow = scaleWidth*4
  guard let destData = malloc(scaleHeight*destBytesPerRow) else {
    print("Error: out of memory")
    return nil
  }
  var destBuffer = vImage_Buffer(data: destData,
                                 height: vImagePixelCount(scaleHeight),
                                 width: vImagePixelCount(scaleWidth),
                                 rowBytes: destBytesPerRow)

  let error = vImageScale_ARGB8888(&srcBuffer, &destBuffer, nil, vImage_Flags(0))
  CVPixelBufferUnlockBaseAddress(srcPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
  if error != kvImageNoError {
    print("Error:", error)
    free(destData)
    return nil
  }

  let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
    if let ptr = ptr {
      free(UnsafeMutableRawPointer(mutating: ptr))
    }
  }

  let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
  var dstPixelBuffer: CVPixelBuffer?
  let status = CVPixelBufferCreateWithBytes(nil, scaleWidth, scaleHeight,
                                            pixelFormat, destData,
                                            destBytesPerRow, releaseCallback,
                                            nil, nil, &dstPixelBuffer)
  if status != kCVReturnSuccess {
    print("Error: could not create new pixel buffer")
    free(destData)
    return nil
  }
  return dstPixelBuffer
}

/**
 Resizes a CVPixelBuffer to a new width and height.
*/
public func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer,
                              width: Int, height: Int) -> CVPixelBuffer? {
  return resizePixelBuffer(pixelBuffer, cropX: 0, cropY: 0,
                           cropWidth: CVPixelBufferGetWidth(pixelBuffer),
                           cropHeight: CVPixelBufferGetHeight(pixelBuffer),
                           scaleWidth: width, scaleHeight: height)
}

/**
 Resizes a CVPixelBuffer to a new width and height.
*/
public func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer,
                              width: Int, height: Int,
                              output: CVPixelBuffer, context: CIContext) {
  let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
  let sx = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
  let sy = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
  let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
  let scaledImage = ciImage.transformed(by: scaleTransform)
  context.render(scaledImage, to: output)
}

public func scaledImageArrayFromPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> MultiArray<Float32>? {
    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        return nil
    }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    var array = MultiArray<Float32>.init(shape: [3, height, width])
    
    for y in 0..<height {
        let row = y * bytesPerRow
        for x in 0..<width {
            let index = x * 4 + row
            let b = Float32(buffer[index]) / 255.0
            let g = Float32(buffer[index + 1]) / 255.0
            let r = Float32(buffer[index + 2]) / 255.0
            array[0, y, x] = r
            array[1, y, x] = g
            array[2, y, x] = b
        }
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    return array
}

public func scaledGrayImageArrayFromPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> MultiArray<Float32>? {
    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        return nil
    }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    var array = MultiArray<Float32>.init(shape: [3, height, width])
    
    for y in 0..<height {
        let row = y * bytesPerRow
        for x in 0..<width {
            let index = x * 4 + row
            let b = Float32(buffer[index])
            let g = Float32(buffer[index + 1])
            let r = Float32(buffer[index + 2])
            let gray = (0.2126 * r + 0.7152 * g + 0.0722 * b) / Float32(255.0)
            array[0, y, x] = gray
            array[1, y, x] = gray
            array[2, y, x] = gray
        }
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    return array
}
