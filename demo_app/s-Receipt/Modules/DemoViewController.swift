//
//  DemoViewController.swift
//  s-Receipt
//
//  Created by Filip Gulan on 18/04/2018.
//  Copyright © 2018 EuroNeuro. All rights reserved.
//

import UIKit

class DemoViewController: UIViewController {
    
    @IBOutlet private weak var logoImageView: UIImageView!
    @IBOutlet private weak var scanButton: UIButton!

    override func viewDidLoad() {
        super.viewDidLoad()
        scanButton.alpha = 0
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.logoImageView.rotate(duration: 2)
            UIView.animate(withDuration: 2) {
                self.scanButton.alpha = 1.0
            }
        }
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    @IBAction private func startScanningActionHandler() {
        let captureVC = storyboard?.instantiateViewController(withIdentifier: "CaptureViewController")
        present(captureVC!, animated: true, completion: nil)
    }    
}
