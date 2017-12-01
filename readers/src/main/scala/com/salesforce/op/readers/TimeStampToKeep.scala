/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import enumeratum._

sealed abstract class TimeStampToKeep extends EnumEntry with Serializable

object TimeStampToKeep extends Enum[TimeStampToKeep] {
  val values = findValues
  case object Min extends TimeStampToKeep
  case object Max extends TimeStampToKeep
  case object Random extends TimeStampToKeep
}
