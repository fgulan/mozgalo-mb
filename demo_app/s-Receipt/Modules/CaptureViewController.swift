import UIKit
import AVFoundation

class CaptureViewController: UIViewController {
    
    @IBOutlet private weak var videoPreview: UIView!
    @IBOutlet private weak var timeLabel: UILabel!
    @IBOutlet private weak var classificationLabel: UILabel!
    @IBOutlet private weak var activityIndicator: UIActivityIndicatorView!
    
    private var videoCapture: VideoCapture!
    private let receiptDetector = ReceiptDetector()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setUpCamera()
    }
    
    @IBAction private func closeButtonActionHandler() {
        dismiss(animated: true)
    }
    
    // MARK: - Initialization
    
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 60
        videoCapture.setUp(sessionPreset: .medium) { [weak self] success in
            
            guard
                let `self` = self,
                let previewLayer = self.videoCapture.previewLayer,
                success else { return }
            
            self.videoPreview.layer.addSublayer(previewLayer)
            self.resizePreviewLayer()
            self.videoCapture.start()
            self.activityIndicator.stopAnimating()
        }
    }
    
    // MARK: - UI
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
    
    func predictReceipt(pixelBuffer: CVPixelBuffer) {
        receiptDetector.recognize(pixelBuffer) { [weak self] result in
            guard let `self` = self else { return }
            
            switch result {
            case .success(let predictions):
                self.classificationLabel.text = self._stringFromPredictions(predictions)
            case .error(let message):
                self.classificationLabel.text = message
            }
            
            self.timeLabel.text = String(format: "%.0f FPS", self.receiptDetector.currentFPS)
        }
    }
    
    private func _stringFromPredictions(_ predictions: [Prediction]) -> String {
//        return "\(_stringFromPrediction(predictions[0]))\n\(_stringFromPrediction(predictions[1]))"
        return "\(predictions[0].receipt)"
    }
    
    private func _stringFromPrediction(_ prediction: Prediction) -> String {
        return String(format: "%@ - %.3lf", prediction.receipt, prediction.prob)
    }
}

extension CaptureViewController: VideoCaptureDelegate {
    
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        
        if let pixelBuffer = pixelBuffer {
            predictReceipt(pixelBuffer: pixelBuffer)
        }
    }
}
