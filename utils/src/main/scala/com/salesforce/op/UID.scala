/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import java.util.concurrent.atomic.AtomicInteger

import org.apache.spark.ml.util.Identifiable

import scala.reflect._
import scala.reflect.runtime.universe._

/**
 * Unique Identifier (UID) generator
 */
case object UID {

  /**
   * Returns a UID that concatenates: classOf[T].getSimpleName + "_" + 12 hex chars.
   * @tparam T type T with a ClassTag
   * @return UID
   */
  def apply[T: ClassTag]: String = apply(classTag[T].runtimeClass)

  /**
   * Returns a UID that concatenates: klazz.getSimpleName + "_" + 12 hex chars.
   * @param klazz class instance
   * @return UID
   */
  def apply(klazz: Class[_]): String = apply(klazz.getSimpleName.stripSuffix("$"))

  /**
   * Returns a UID that concatenates: prefix + "_" + 12 hex chars.
   * @param prefix uid prefix
   * @return UID
   */
  def apply(prefix: String): String = makeUID(prefix, isSequential = true)

  /**
   * Parses UID from string
   * @param uid a UID that concatenates: prefix + "_" + 12 hex chars
   * @throws IllegalArgumentException in case an invalid UID is specified
   * @return (prefix, suffix) tuple
   */
  def fromString(uid: String): (String, String) = {
    try { uid.split("_") match { case Array(prefix, suffix) => prefix -> suffix } }
    catch {
      case _: Exception => throw new IllegalArgumentException(s"Invalid UID: $uid")
    }
  }

  /**
   * Resets the UID counter back to specified count.
   * Can be useful when generating workflows programmatically, but the UIDs needs to be the same.
   *
   * @param v reset count to value v (default: 0)
   * NOTE: Don't use this method unless you know what you are doing.
   */
  def reset(v: Int = 0): this.type = {
    counter.set(v)
    this
  }

  /**
   * Gets current UID count
   *
   * @return UID counter value
   */
  def count(): Int = counter.get()

  private val counter = new AtomicInteger(0)

  private def makeUID(prefix: String, isSequential: Boolean): String = {
    if (isSequential) {
      val id = counter.incrementAndGet()
      String.format(s"${prefix}_%12s", Integer.toHexString(id)).replace(" ", "0")
    } else {
      Identifiable.randomUID(prefix = prefix)
    }
  }

}
