/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import enumeratum._


sealed trait CutOffTimeType extends EnumEntry with Serializable

object CutOffTimeTypes extends Enum[CutOffTimeType] {
  val values = findValues
  case object UnixEpoch extends CutOffTimeType
  case object DaysAgo extends CutOffTimeType
  case object WeeksAgo extends CutOffTimeType
  case object DDMMYYYY extends CutOffTimeType
  case object NoCutoff extends CutOffTimeType
}

