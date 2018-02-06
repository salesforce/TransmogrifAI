/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.types


class TestPrivateType private[types](val x: Int, val y: Int) {
  private def this() = this(0, 0)
  def this(v: Int) = this(v, v)
  def this(x: Int, y: Int, z: Int) = this(x + z, y + z)
}

object TestPrivateType {
  def apply(x: Int, y: Int): TestPrivateType = new TestPrivateType(x, y)
}
