/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.readers

import com.salesforce.op.features.OPFeature
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.FeatureGeneratorStage
import com.salesforce.op.{OpParams, ReaderParams}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.reflect.runtime.universe.WeakTypeTag


private[readers] trait ReaderType[T] extends Serializable {

  /**
   * Reader type tag
   */
  implicit val wtt: WeakTypeTag[T]

  /**
   * Full reader input type name
   *
   * @return full input type name
   */
  final def fullTypeName: String = wtt.tpe.toString

  /**
   * Short reader input type name
   *
   * @return short reader input type name
   */
  final def typeName: String = fullTypeName.split("\\.").last

  /**
   * Default method for extracting this reader's parameters from readerParams in [[OpParams]]
   *
   * @param opParams contains map of reader type to ReaderParams instances
   * @return ReaderParams instance if it exists
   */
  final def getReaderParams(opParams: OpParams): Option[ReaderParams] = opParams.readerParams.get(this.typeName)

}


private[readers] trait ReaderKey[T] extends Serializable {

  /**
   * Function for extracting key from a record
   * @return key string
   */
  def key: T => String

}

object ReaderKey {

  /**
   * Random key function
   *
   * @return a random key value
   */
  def randomKey[T](t: T): String = util.Random.nextLong().toString

}


trait Reader[T] extends ReaderType[T] {

  /**
   * All the reader's sub readers (used in joins)
   * @return sub readers
   */
  def subReaders: Seq[DataReader[_]]

  /**
   * Outer join
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final def outerJoin[U](other: DataReader[U], joinKeys: JoinKeys = JoinKeys()): JoinedDataReader[T, U] =
    join(other, joinType = JoinTypes.Outer, joinKeys)

  /**
   * Left Outer join
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final def leftOuterJoin[U](other: DataReader[U], joinKeys: JoinKeys = JoinKeys()): JoinedDataReader[T, U] =
    join(other, joinType = JoinTypes.LeftOuter, joinKeys)

  /**
   * Inner join
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final def innerJoin[U](other: DataReader[U], joinKeys: JoinKeys = JoinKeys()): JoinedDataReader[T, U] =
    join(other, joinType = JoinTypes.Inner, joinKeys)

  /**
   * Join readers
   *
   * @param other    reader from right side of join
   * @param joinKeys join keys to use
   * @param joinType type of join to perform
   * @tparam U Type of data read by right data reader
   * @return joined reader
   */
  final protected def join[U](
    other: DataReader[U],
    joinType: JoinType,
    joinKeys: JoinKeys = JoinKeys()
  ): JoinedDataReader[T, U] = {
    val joinedReader =
      new JoinedDataReader[T, U](leftReader = this, rightReader = other, joinKeys = joinKeys, joinType = joinType)
    assert(joinedReader.leftReader.subReaders
      .forall(r => r.fullTypeName != joinedReader.rightReader.fullTypeName),
      "All joins must be for readers of different objects - self joins are not supported"
    )
    joinedReader
  }

  /**
   * Generate the dataframe that will be used in the OpPipeline calling this method
   *
   * @param rawFeatures features to generate from the dataset read in by this reader
   * @param opParams    op parameters
   * @param spark       spark instance to do the reading and conversion from RDD to Dataframe
   * @return A dataframe containing columns with all of the raw input features expected by the pipeline
   */
  def generateDataFrame(rawFeatures: Array[OPFeature], opParams: OpParams)(implicit spark: SparkSession): DataFrame

  protected[op] def getGenStage[I](f: OPFeature): FeatureGeneratorStage[I, _ <: FeatureType] = {
    f.originStage match {
      case fg: FeatureGeneratorStage[_, _] => fg.asInstanceOf[FeatureGeneratorStage[I, _ <: FeatureType]]
      case _ => throw new IllegalArgumentException(
        s"All raw features must have origin stage of type ${
          classOf[FeatureGeneratorStage[I, _ <: FeatureType]].getSimpleName
        }"
      )
    }
  }
}
