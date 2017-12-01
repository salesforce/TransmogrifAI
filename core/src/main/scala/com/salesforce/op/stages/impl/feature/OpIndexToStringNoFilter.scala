/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.stages.sparkwrappers.specific.OpTransformerWrapper
import org.apache.spark.annotation.Since
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.param.{Param, StringArrayParam}

/**
 * A transformer that maps a feature of indices back to a new feature of corresponding text values.
 * The index-string mapping is either from the ML attributes of the input feature,
 * or from user-supplied labels (which take precedence over ML attributes).
 *
 * @see [[OpStringIndexerNoFilter]] for converting text into indices
 */
class OpIndexToStringNoFilter(uid: String = UID[OpIndexToStringNoFilter])
  extends UnaryTransformer[RealNN, Text](operationName = "idx2str", uid = uid) with SaveOthersParams {

  final val labels: StringArrayParam = new StringArrayParam(this, "labels",
    "Optional array of labels specifying index-string mapping." +
      " If not provided or if empty, then metadata from inputCol is used instead.")

  final def getLabels: Array[String] = $(labels)

  final def setLabels(labelsIn: Array[String]): this.type = set(labels, labelsIn)

  setDefault(unseenName, OpIndexToStringNoFilter.unseenDefault)

  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN) => Text = {
    (input: RealNN) => {
      val inputColSchema = getInputSchema()(in1.name)
      // If the labels array is empty use column metadata
      val lbls = $(labels)
      val unseen = $(unseenName)
      val values = if (!isDefined(labels) || lbls.isEmpty) {
        Attribute.fromStructField(inputColSchema)
          .asInstanceOf[NominalAttribute].values.get
      } else {
        lbls
      }
      val idx = input.value.get.toInt
      if (0 <= idx && idx < values.length) {
        values(idx).toText
      } else {
        unseen.toText
      }
    }
  }
}

object OpIndexToStringNoFilter {
  val unseenDefault: String = "UnseenIndex"
}

