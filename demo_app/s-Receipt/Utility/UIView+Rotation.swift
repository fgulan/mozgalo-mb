//
//  UIView+Rotation.swift
//  s-Receipt
//
//  Created by Filip Gulan on 05/06/2018.
//  Copyright © 2018 EuroNeuro. All rights reserved.
//

import UIKit

extension UIView {
    
    private static let kRotationAnimationKey = "rotationAnimationKey"
    
    func rotate(duration: Double = 1, repeatCount: Float = 1) {
        if layer.animation(forKey: UIView.kRotationAnimationKey) == nil {
            let rotationAnimation = CABasicAnimation(keyPath: "transform.rotation")
            
            rotationAnimation.fromValue = 0.0
            rotationAnimation.toValue = Float.pi * 2.0
            rotationAnimation.duration = duration
            rotationAnimation.repeatCount = repeatCount
            
            layer.add(rotationAnimation, forKey: UIView.kRotationAnimationKey)
        }
    }
    
    func stopRotating() {
        if layer.animation(forKey: UIView.kRotationAnimationKey) != nil {
            layer.removeAnimation(forKey: UIView.kRotationAnimationKey)
        }
    }
}
