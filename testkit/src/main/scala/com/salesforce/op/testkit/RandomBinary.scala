/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types.Binary

import scala.util.Random

/**
 * Generates Binary FeatureType, that is, a stream of Booleans wrapped as FeatureType
 *
 * @param probabilityOfSuccess the ratio of true values in the stream
 *
 */
case class RandomBinary
(
  probabilityOfSuccess: Double
) extends StandardRandomData[Binary](
  sourceOfData = RandomStream.ofBitOptions(probabilityOfSuccess)
) with ProbabilityOfEmpty
