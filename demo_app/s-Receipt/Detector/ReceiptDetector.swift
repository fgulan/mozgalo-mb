//
//  ReceiptDetector.swift
//  s-Receipt
//
//  Created by Filip Gulan on 04/05/2018.
//  Copyright © 2018 EuroNeuro. All rights reserved.
//

import UIKit
import CoreML

typealias Prediction = (receipt: String, prob: Double)

final class ReceiptDetector {
    
    private static let _inputHeight = 370
    private static let _inputWidth = 400
    
    private let _inputShape = [3, ReceiptDetector._inputHeight, ReceiptDetector._inputWidth]
    private let _model = MacroBling()
    private var _startTimes: [CFTimeInterval] = []
    private var _framesDone = 0
    private var _frameCapturingStartTime = CACurrentMediaTime()
    private let _semaphore = DispatchSemaphore(value: 2)
    
    private(set) var currentFPS: Double = 0.0

    func recognize(_ pixelBuffer: CVPixelBuffer, completion: @escaping (Result<[Prediction]>) -> ()) {
        _semaphore.wait()
        self._startTimes.append(CACurrentMediaTime())
        DispatchQueue.global().async {
            let bufferWidth = CVPixelBufferGetWidth(pixelBuffer)
            let bufferHeight = CVPixelBufferGetHeight(pixelBuffer)
            let pixelBufferResized =
                resizePixelBuffer(pixelBuffer, cropX: 0, cropY: 0,
                                  cropWidth: bufferWidth,
                                  cropHeight: Int(0.5 * Double(bufferHeight)),
                                  scaleWidth: ReceiptDetector._inputWidth,
                                  scaleHeight: ReceiptDetector._inputHeight)
            
            let result: Result<[Prediction]>
            if let buffer = pixelBufferResized,
                let input = scaledGrayImageArrayFromPixelBuffer(buffer),
                let outputs = try? self._model.prediction(_0: input.array) {
                result = .success(top(2, outputs._117))
            } else {
                result = .error("Prediction error!")
            }
            
            DispatchQueue.main.async {
                self.currentFPS = self._calculateFps()
                completion(result)
                self._semaphore.signal()
            }
        }
    }
    
    private func _calculateFps() -> Double {
        _framesDone += 1
        let frameCapturingElapsed = CACurrentMediaTime() - _frameCapturingStartTime
        let currentFpsDelivered = Double(_framesDone) / frameCapturingElapsed
        if frameCapturingElapsed > 1 {
            _framesDone = 0
            _frameCapturingStartTime = CACurrentMediaTime()
        }
        return currentFpsDelivered
    }
}
