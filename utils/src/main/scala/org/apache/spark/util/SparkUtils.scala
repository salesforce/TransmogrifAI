/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.util


object SparkUtils {

  /** Preferred alternative to Class.forName(className) */
  def classForName(name: String): Class[_] = Utils.classForName(name)

}
