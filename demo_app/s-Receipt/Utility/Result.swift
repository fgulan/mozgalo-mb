//
//  Result.swift
//  s-Receipt
//
//  Created by Filip Gulan on 04/05/2018.
//  Copyright © 2018 EuroNeuro. All rights reserved.
//

import Foundation

enum Result<T> {
    case success(T)
    case error(String)
}
