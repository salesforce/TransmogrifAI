/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import enumeratum._

sealed abstract class JoinType(val sparkJoinName: String) extends EnumEntry with Serializable

object JoinTypes extends Enum[JoinType] {
  val values = findValues
  case object Outer extends JoinType("outer")
  case object LeftOuter extends JoinType("left_outer")
  case object Inner extends JoinType("inner")
}
