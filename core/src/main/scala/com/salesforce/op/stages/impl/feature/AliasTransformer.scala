/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.runtime.universe.TypeTag

/**
 * No-op (identity) alias feature transformer allowing renaming features
 * without applying a transformation on values.
 *
 * @param name desired feature name
 * @param uid uid for instance
 * @param tti  type tag for input and output
 * @param ttov type tag for output value
 * @tparam I feature type
 */
class AliasTransformer[I <: FeatureType](val name: String, uid: String = UID[AliasTransformer[_]])
  (implicit tti: TypeTag[I], ttov: TypeTag[I#Value])
  extends UnaryTransformer[I, I](operationName = "alias", uid = uid)(tti = tti, tto = tti, ttov = ttov) {

  override def transformFn: I => I = identity
  override def outputName: String = name
  override def transform(dataset: Dataset[_]): DataFrame = {
    val newSchema = setInputSchema(dataset.schema).transformSchema(dataset.schema)
    val meta = newSchema(in1.name).metadata
    dataset.select(col("*"), col(in1.name).as(outputName, meta))
  }

}
