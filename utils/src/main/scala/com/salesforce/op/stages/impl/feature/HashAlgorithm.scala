/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import enumeratum._


/**
 * Hashing Algorithms
 */
sealed trait HashAlgorithm extends EnumEntry with Serializable

object HashAlgorithm extends Enum[HashAlgorithm] {
  val values = findValues
  case object MurMur3 extends HashAlgorithm
  case object Native extends HashAlgorithm
}
