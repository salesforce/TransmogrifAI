/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import java.io.{ByteArrayOutputStream, File, StringReader}

import com.salesforce.op.OpWorkflowRunType
import com.salesforce.op.test.TestCommon
import org.scalactic.source
import org.scalatest.{Assertion, Assertions, BeforeAndAfter, FlatSpec}
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.language.postfixOps

/**
 * Test for generator operations
 */
class CliTestBase extends FlatSpec with TestCommon with Assertions with BeforeAndAfter {
  CommandParser.AUTO_ENABLED = true

  protected val ProjectName = "CliGeneratedTestProject"
  val log = LoggerFactory.getLogger("cli-test")

  trait Outcome
  case class Crashed(msg: String, code: Int) extends Throwable with Outcome {
    override def toString: String = s"Crashed($msg, $code)"
  }
  case object Succeeded extends Outcome
  case class Result(outcome: Outcome, out: String, err: String)

  def assertResult(r: Result, expected: Outcome)(implicit pos: source.Position): Assertion = {
    def logMsg = s"out=\n${r.out}\nerr=\n${r.err}"
    r.outcome match {
      case v if v == expected =>
        log.info(logMsg)
        succeed
      case bad@Crashed(msg, code) =>
        log.error(logMsg)
        log.error("Crash stack trace: ", bad: Throwable)
        fail(s"Crashed: code=$code, msg='$msg'")
      case v =>
        log.error(logMsg)
        fail(s"Unexpected result: $v")
    }
  }

  class Sut(input: String = "") extends CliExec {
    override protected val DEBUG = true
    override def quit(errorMsg: String, code: Int = -1): Nothing = {
      throw Crashed(errorMsg, code)
    }

    def run(args: String*): Result = {
      val out = new ByteArrayOutputStream
      val err = new ByteArrayOutputStream

      val outcome =
        Console.withIn(new StringReader(input)) {
          Console.withOut(out) {
            Console.withErr(err) {
              try {
                super.main(args.toArray)
                Succeeded
              } catch {
                case oops: Crashed =>
                  Console.err.println(oops.msg)
                  oops
              }
            }
          }
        }
      Result(outcome, out.toString, err.toString)
    }
  }

  before { new Sut().delete(new File(ProjectName)) }

  after { new Sut().delete(new File(ProjectName)) }

  val expectedSourceFiles = "Features.scala" :: s"$ProjectName.scala"::Nil

  val projectDir = new File(ProjectName.toLowerCase)

  def checkAvroFile(source: File): Unit = {
    val avroFile = new File(projectDir, s"src/main/avro/${source.getName}")
    avroFile should exist
    Source.fromFile(avroFile).getLines.mkString("\n") shouldBe
      Source.fromFile(source).getLines.mkString("\n")
  }

  def checkScalaFiles(shouldNotContain: String): Unit = {
    val srcDir = new File(projectDir, "src/main/scala/com/salesforce/app")
    srcDir should exist

    for {
      filename <- expectedSourceFiles
      file = new File(srcDir, filename)
      _ = file should exist
      contents = Source.fromFile(file).getLines.zipWithIndex
      (line, n) <- contents
    } {
      withClue(s"File $file, line $n\n$line") {
        line.contains(shouldNotContain) shouldBe false
      }
    }
  }

  def findFile(relPath: String): String = {
    Option(new File(relPath)) filter (_.exists) orElse
    Option(new File(new File(".."), relPath)) filter (_.exists) getOrElse {
      throw new UnsupportedOperationException(
        s"Could not find file $relPath, current is ${new File(".").getAbsolutePath}")
    } getAbsolutePath
  }

  protected lazy val TestAvsc: String = findFile("test-data/PassengerDataAll.avsc")
  protected lazy val TestCsvHeadless: String = findFile("test-data/PassengerDataAll.csv")
  protected lazy val TestSmallCsvWithHeaders: String = findFile("test-data/PassengerDataWithHeader.csv")
  protected lazy val TestBigCsvWithHeaders: String = findFile("test-data/PassengerDataAllWithHeader.csv")
  protected lazy val AvcsSchema: String = findFile("templates/simple/src/main/avro/Passenger.avsc")
  protected lazy val AnswersFile: String = findFile("cli/passengers.answers")

  protected def appRuntimeArgs(runType: OpWorkflowRunType) =
    s"--run-type=${runType.toString.toLowerCase} --model-location=/tmp/titanic-model " +
    s"--read-location Passenger=$TestCsvHeadless"
}
