/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import org.apache.spark.sql.types.{ArrayType, DataType, MapType}

import scala.reflect.runtime.universe._

/**
 * Abstract base class for a set of transformer wrappers that allow unary transformers between non-collection types
 * to be used on collection types. For example, we can use a UnaryLambdaTransformer[Email, Integer] on a map's values,
 * creating a UnaryLambdaTransformer[EmailMap, IntegralMap]. This base class will be inherited by concrete classes for
 * OPMaps, OPList, and OPSets (in order to enforce not allowing these collection types to be transformed into
 * each other, eg. no MultiPickList to RealMap transformations).
 *
 * The OP type hierarchy does not allow direct type checking of such transformer wrappers (eg. Real#Value is
 * Option[Double] and RealMap#Value is Map[String, Double], so there's no way to enforce that a RealMap can only
 * hold what is contained in a Real) since the types themselves are not created with typetags for performance reasons.
 * However, we can still enforce that operations like building a UnaryLambdaTransformer[RealMap, StringMap] from a
 * UnaryLambdaTransformer[Real, Integer] is not possible by using the Spark types in validateTypes.
 *
 *
 * @param transformer   UnaryTransformer on non-collection types that we wish to use on collection types
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I            input feature type for supplied non-collection transformer
 * @tparam O            output feature type for supplied non-collection transformer
 * @tparam ICol         input feature type for desired collection transformer
 * @tparam OCol         output feature type for desired collection transformer
 */
sealed abstract class OPCollectionTransformer[I <: FeatureType,
O <: FeatureType, ICol <: OPCollection, OCol <: OPCollection]
(
  val transformer: UnaryTransformer[I, O],
  operationName: String,
  uid: String
)(implicit tti: TypeTag[ICol], tto: TypeTag[OCol], ttov: TypeTag[OCol#Value])
  extends UnaryTransformer[ICol, OCol](operationName = operationName, uid = uid) {

  // Perform input types validation using Spark data type and throw an exception early on
  requireValidateTypes()

  val inFactory = FeatureTypeSparkConverter[I]()(transformer.tti)
  val outFactory = FeatureTypeFactory[OCol]()
  val outEmpty = FeatureTypeDefaults.default[OCol]

  def transformFn: ICol => OCol = in => if (in.isEmpty) outEmpty else outFactory.newInstance(doTransform(in))

  /**
   * Function that transforms the relevant entry of the collection based on the supplied transformer. OPLists and
   * OPSets have their entries transformed, and OPMaps have just their values transformed.
   */
  protected def doTransform: ICol => Any

  /**
   * Function that takes the values contained in our FeatureTypes and converts them to/from types that the supplied
   * UnaryTransformer operates on and applies the unary transformation. Note that this does not do any checks on the
   * output, so if the unary transformer function produces a null, then we leave that in the map
   * (Map[String, Long] can still contain eg. ("a" -> null))
   *
   * @param value input value contained in type ICol (eg. if ICol is TextMap, then value is Map[String, String])
   * @return transformed output value contained
   */
  protected def transformValue(value: Any): Any = {
    val iVal: I = inFactory.fromSpark(value)
    val oVal: O = transformer.transformFn(iVal)
    FeatureTypeSparkConverter.toSpark(oVal)
  }

  /**
   * Function performs input types validation using Spark data types and throws an exception early on
   * (during workflow generation). It also ensures that collection types cannot be used in the supplied
   * transformer since l and r in validateTypes(l, r) are fully unwrapped.
   *
   * @throws IllegalArgumentException
   */
  private def requireValidateTypes(): Unit = {
    def validateTypes(l: DataType, r: DataType): Boolean = (l, r) match {
      case (ArrayType(fromElement, _), ArrayType(toElement, _)) => validateTypes(fromElement, toElement)
      case (MapType(_, fromValue, _), MapType(_, toValue, _)) => validateTypes(fromValue, toValue)
      case (fromDataType, toDataType)
        if fromDataType == FeatureSparkTypes.sparkTypeOf[I](transformer.tti) &&
          toDataType == FeatureSparkTypes.sparkTypeOf[O](transformer.tto) => true
      case _ => false
    }

    require(
      validateTypes(FeatureSparkTypes.sparkTypeOf[ICol], FeatureSparkTypes.sparkTypeOf[OCol]),
      s"Type ${tti.tpe.typeSymbol.name} is not convertible to ${tto.tpe.typeSymbol.name} with given " +
        s"${transformer.getClass.getSimpleName}[${transformer.tti.tpe.typeSymbol.name}, " +
        s"${transformer.tto.tpe.typeSymbol.name}]"
    )
  }
}


/**
 * Concrete class to take a UnaryTransformer from non-collection types to a UnaryTransformer between corresponding
 * OPMap types. Types are checked so that eg. UnaryLambdaTransformer[Real, Text] should be able to be transformed into
 * UnaryLambdaTransformer[RealMap, TextMap], but not UnaryLambdaTransformer[RealMap, IntegralMap].
 *
 * @param transformer   UnaryTransformer on non-collection types that we wish to use on OPMap types
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I            input feature type for supplied non-collection transformer
 * @tparam O            output feature type for supplied non-collection transformer
 * @tparam IMap         input feature type for desired OPMap transformer
 * @tparam OMap         output feature type for desired OPMap transformer
 */
private[op] class OPMapTransformer[I <: FeatureType, O <: FeatureType, IMap <: OPMap[_], OMap <: OPMap[_]]
(
  transformer: UnaryTransformer[I, O],
  operationName: String,
  uid: String = UID[OPMapTransformer[_, _, _, _]]
)(implicit tti: TypeTag[IMap], tto: TypeTag[OMap], ttov: TypeTag[OMap#Value])
  extends OPCollectionTransformer[I, O, IMap, OMap](
    transformer = transformer, operationName = operationName, uid = uid) {
  def doTransform: IMap => Any = _.value.map { case (key, value) => key -> transformValue(value) }
}

/**
 * Concrete class to take a UnaryTransformer from non-collection types to a UnaryTransformer between corresponding
 * OPSet types. Types are checked so that eg. UnaryLambdaTransformer[Real, Text] should be able to be transformed into
 * UnaryLambdaTransformer[MultiPickList, MultiPickList], but UnaryLambdaTransformer[Real, Real] should not be able to.
 *
 * @param transformer   UnaryTransformer on non-collection types that we wish to use on OPSet types
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I            input feature type for supplied non-collection transformer
 * @tparam O            output feature type for supplied non-collection transformer
 * @tparam ISet         input feature type for desired OPSet transformer
 * @tparam OSet         output feature type for desired OPSet transformer
 */
private[op] class OPSetTransformer[I <: FeatureType, O <: FeatureType, ISet <: OPSet[_], OSet <: OPSet[_]]
(
  transformer: UnaryTransformer[I, O],
  operationName: String,
  uid: String = UID[OPSetTransformer[_, _, _, _]]
)(implicit tti: TypeTag[ISet], tto: TypeTag[OSet], ttov: TypeTag[OSet#Value])
  extends OPCollectionTransformer[I, O, ISet, OSet](
    transformer = transformer, operationName = operationName, uid = uid) {
  def doTransform: ISet => Any = _.value.map(transformValue)
}

/**
 * Concrete class to take a UnaryTransformer from non-collection types to a UnaryTransformer between corresponding
 * OPList types. Types are checked so that eg. UnaryLambdaTransformer[Text, Text] should be able to be transformed into
 * UnaryLambdaTransformer[TextList, TextList], but UnaryLambdaTransformer[Text, Real] should not be able to.
 *
 * @param transformer   UnaryTransformer on non-collection types that we wish to use on OPList types
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I            input feature type for supplied non-collection transformer
 * @tparam O            output feature type for supplied non-collection transformer
 * @tparam IList        input feature type for desired OPList transformer
 * @tparam OList        output feature type for desired OPList transformer
 */
private[op] class OPListTransformer[I <: FeatureType, O <: FeatureType, IList <: OPList[_], OList <: OPList[_]]
(
  transformer: UnaryTransformer[I, O],
  operationName: String,
  uid: String = UID[OPListTransformer[_, _, _, _]]
)(implicit tti: TypeTag[IList], tto: TypeTag[OList], ttov: TypeTag[OList#Value])
  extends OPCollectionTransformer[I, O, IList, OList](
    transformer = transformer, operationName = operationName, uid = uid) {
  def doTransform: IList => Any = _.value.map(transformValue)
}
