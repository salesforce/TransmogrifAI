package com.salesforce.op.stages.base

import com.salesforce.op.features.types.FeatureType

import scala.reflect.runtime.universe.TypeTag

/**
 * @author ksuchanek
 * @since 214
 */
trait LambdaTransformer[O <: FeatureType, F] {
  val tto: TypeTag[O]
  val ttov: TypeTag[O#Value]
  val transformFn: F
  val lambdaCtorArgs:Array[_]
}
