/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

/**
 * Represents one of the three kinds of machine learning problems.
 */
sealed trait ProblemKind

object ProblemKind {

  case object BinaryClassification extends ProblemKind
  case object MultiClassification extends ProblemKind
  case object Regression extends ProblemKind

  /**
   * Holds every possible [[ProblemKind]]
   */
  val values: List[ProblemKind] = List(BinaryClassification, MultiClassification, Regression)

  /**
   * Ask the user to identify the problem kind (because we were unable to identify it)
   *
   * @param responseFieldSchema The avro field for the response field, we display the name to the user
   * @param available What [[ProblemKind]]s are available for us to prompt the user with
   * @return The [[ProblemKind]] the user picked
   */
  private[gen] def askKind(
    ops: Ops,
    responseFieldSchema: AvroField,
    available: List[ProblemKind] = values): ProblemKind = {
    val options = Map(
      Regression -> List("regress", "regression"),
      BinaryClassification -> List("binclass", "binary classification"),
      MultiClassification -> List("multiclass", "multi classification")
    ).filter {
      case (kind, _) => available.contains(kind)
    }
    ops.ask(
      s"Cannot infer the kind of problem based on response field ${responseFieldSchema.name}.\n" +
        "What kind of problem is this?", options
    )
  }

  /**
   * Build a [[ProblemKind]] from the schema for an [[AvroField]]. May prompt the user if more information is required.
   *
   * @param responseFieldSchema The schema to build from
   * @return The built [[ProblemKind]]
   */
  def from(ops: Ops, responseFieldSchema: AvroField): ProblemKind = {
    if (responseFieldSchema.nullable) {
      Ops.oops(s"Response field '$responseFieldSchema' cannot be nullable")
    }
    responseFieldSchema.problemKind(ops)
  }
}
