/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl

import org.apache.spark.ml.param.ParamMap
import org.scalatest.{Assertions, Matchers}


trait CompareParamGrid extends Matchers with Assertions {

  /**
   * Compare two params grids
   */
  def gridCompare(g1: Array[ParamMap], g2: Array[ParamMap]): Unit = {
    val g1values = g1.toSet[ParamMap].map(_.toSeq.toSet)
    val g2values = g2.toSet[ParamMap].map(_.toSeq.toSet)
    matchTwoSets(g1values, g2values)
  }

  private def matchTwoSets[T](actual: Set[T], expected: Set[T]): Unit = {
    def stringify(set: Set[T]): String = {
      val list = set.toList
      val chunk = list take 10
      val strings = chunk.map(_.toString).sorted
      if (list.size > chunk.size) strings.mkString else strings.mkString + ",..."
    }
    val missing = stringify(expected -- actual)
    val extra = stringify(actual -- expected)
    withClue(s"Missing:\n $missing\nExtra:\n$extra") {
      actual shouldBe expected
    }
  }
}
