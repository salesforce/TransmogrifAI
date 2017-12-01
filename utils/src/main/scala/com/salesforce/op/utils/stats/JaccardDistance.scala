/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.stats

/**
 * Calculates the Jaccard Distance between two sets.
 * If both inputs are empty, Jaccard Distance is defined as 1.0
 */
object JaccardDistance {

  /**
   * Calculates the Jaccard Distance between two sets.
   * If both inputs are empty, Jaccard Distance is defined as 1.0
   *
   * @param s1 first set
   * @param s2 second set
   * @tparam A set type
   * @return Jaccard Distance
   */
  def apply[A](s1: Set[A], s2: Set[A]): Double = {
    val intersectSize = s1.intersect(s2).size
    val unionSize = s1.size + s2.size - intersectSize
    if (unionSize == 0) 1.0 else intersectSize.toDouble / unionSize
  }

}
