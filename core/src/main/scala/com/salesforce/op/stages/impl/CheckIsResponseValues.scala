/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl

import com.salesforce.op.features.TransientFeature

/**
 * Factory to check if the first input is a response and the second one is not
 */
private[op] case object CheckIsResponseValues {
  def apply(in1: TransientFeature, in2: TransientFeature): Unit = {
    if (!in1.isResponse) {
      throw new IllegalArgumentException("The numeric 'label' feature should be a response feature.")
    }
    if (in2.isResponse) {
      throw new IllegalArgumentException("The feature vector should not contain any response features.")
    }
  }
}
