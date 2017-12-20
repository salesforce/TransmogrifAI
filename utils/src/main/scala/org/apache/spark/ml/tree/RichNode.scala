/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.tree

object RichNode {

  /**
   * Enrichment functions for Decision Tree node
   * @param node Decision Tree node
   */
  implicit class RichNode(val node: Node) extends AnyVal {

    private implicit def splitToArray(s: Split): Array[Double] = s match {
      case c: ContinuousSplit => Array(c.threshold)
      case c: CategoricalSplit => c.leftCategories
    }

    /**
     * Compute Decision tree splits for a given node recursively
     *
     * @return Decision tree splits sorted in ascending order
     */
    def splits: Array[Double] = node match {
      case i: InternalNode => ((i.split: Array[Double]) ++ i.leftChild.splits ++ i.rightChild.splits).distinct.sorted
      case _: LeafNode => Array.empty[Double]
    }

  }

}
