/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op.features.TransientFeature


trait HasIn1 {
  self: OpPipelineStageBase =>
  final protected def in1: TransientFeature = getTransientFeature(0).get
}

trait HasIn2 {
  self: OpPipelineStageBase =>
  final protected def in2: TransientFeature = getTransientFeature(1).get
}

trait HasIn3 {
  self: OpPipelineStageBase =>
  final protected def in3: TransientFeature = getTransientFeature(2).get
}

trait HasIn4 {
  self: OpPipelineStageBase =>
  final protected def in4: TransientFeature = getTransientFeature(3).get
}

trait HasInN {
  self: OpPipelineStageBase =>
  final protected def inN: Array[TransientFeature] = getTransientFeatures()
}

