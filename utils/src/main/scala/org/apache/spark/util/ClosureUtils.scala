/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.util

import scala.util.Try


case object ClosureUtils {

  /**
   * Check if a closure is serializable
   *
   * @return Failure if not serializable
   */
  def checkSerializable(closure: AnyRef): Try[Unit] = Try {
    ClosureCleaner.clean(closure, checkSerializable = true, cleanTransitively = true)
  }

}
