package org.apache.spark.sql.expressions

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.DataType

object SparkUDFFactory {
  /**
   * A public interface to Spark 3's private org.apache.spark.sql.expressions.SparkUserDefinedFunction,
   * replacing Spark's 2.4 UserDefinedFunction case class.
   * @param f             The user defined function as a closure
   * @param dataType      the output Spark DataType
   * @param inputEncoders --
   * @param outputEncoder --
   * @param name          --
   * @param nullable      --
   * @param deterministic -- See Spark code/documentation for those parameters, they're not needed in TMog
   * @return A Spark UserDefinedFunction
   */
  def create(
    f: AnyRef,
    dataType: DataType,
    inputEncoders: Seq[Option[ExpressionEncoder[_]]] = Nil,
    outputEncoder: Option[ExpressionEncoder[_]] = None,
    name: Option[String] = None,
    nullable: Boolean = true,
    deterministic: Boolean = true
  ) : UserDefinedFunction = {
    SparkUserDefinedFunction(
      f = f,
      dataType = dataType,
      inputEncoders = inputEncoders,
      outputEncoder = outputEncoder,
      name = name,
      nullable = nullable,
      deterministic = deterministic
    )
  }
}
