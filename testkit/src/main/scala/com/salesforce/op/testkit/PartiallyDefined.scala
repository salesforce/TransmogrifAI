/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

/**
 * An aspect consisting of something being defined
 */
trait PartiallyDefined {
  /**
   * A flag that tells whether we should skip the value and build an empty instead
   * It's overridden in ProbabilityOfEmpty trait
   *
   * @return true if skip
   */
  def skip: Boolean = false
}
