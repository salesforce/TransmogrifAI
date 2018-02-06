/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml

import com.salesforce.op.stages.OpPipelineStageBase
import org.apache.spark.SparkContext
import org.apache.spark.ml.util.DefaultParamsReader.{Metadata, loadMetadata}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter}

/**
 * Direct wrappers for ml private [[DefaultParamsWriter]] and [[DefaultParamsReader]]
 * needed to read/write Spark stages in OP
 */
case object SparkDefaultParamsReadWrite {

  /**
   * Helper for [[saveMetadata()]] which extracts the JSON to save.
   * This is useful for ensemble models which need to save metadata for many sub-models.
   *
   * @see [[saveMetadata()]] for details on what this includes.
   */
  def getMetadataToSave(stage: OpPipelineStageBase, sc: SparkContext): String =
    DefaultParamsWriter.getMetadataToSave(stage, sc)

  /**
   * Parse metadata JSON string produced by [[DefaultParamsWriter.getMetadataToSave()]].
   * This is a helper function for [[loadMetadata()]].
   *
   * @param metadataStr  JSON string of metadata
   * @param expectedClassName  If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def parseMetadata(jsonStr: String): DefaultParamsReader.Metadata =
    DefaultParamsReader.parseMetadata(jsonStr)

  /**
   * Extract Params from metadata, and set them in the instance.
   * This works if all Params implement [[org.apache.spark.ml.param.Param.jsonDecode()]].
   * TODO: Move to [[Metadata]] method
   */
  def getAndSetParams(stage: OpPipelineStageBase, metadata: DefaultParamsReader.Metadata): Unit =
    DefaultParamsReader.getAndSetParams(stage, metadata)

}
