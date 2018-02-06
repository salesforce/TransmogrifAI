/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.numeric

object Number extends Serializable {
  def isValid(x: Double): Boolean = !x.isNaN && !x.isInfinity
}
