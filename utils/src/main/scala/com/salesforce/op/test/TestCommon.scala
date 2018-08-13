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

package com.salesforce.op.test

import java.io.File

import org.apache.log4j.{Level, LogManager, Logger}
import org.scalatest._

import scala.collection.JavaConverters._
import scala.io.Source
import scala.language.postfixOps
import scala.reflect.{ClassTag, _}

/**
 * Trait with test commons such as Spec and Resource functions
 */
trait TestCommon extends Matchers with Assertions {
  private val _resourceDir = "src/test/resources"

  /**
   * Returns the resource directory path
   */
  protected def resourceDir = _resourceDir

  /**
   * Set logging level for
   */
  def loggingLevel(level: Level): Unit = {
    val loggers = Logger.getRootLogger :: LogManager.getCurrentLoggers.asScala.toList
    loggers.collect { case l: Logger => l }.foreach(_.setLevel(level))
  }

  /**
   * Disable all logging
   */
  def loggingOff(): Unit = loggingLevel(Level.OFF)

  /**
   * Spec name for a class
   */
  case object Spec {
    def apply[T: ClassTag]: String = apply(classTag[T].runtimeClass)
    def apply[T1: ClassTag, T2: ClassTag]: String = apply[T2] + "[" + apply[T1] + "]"
    def apply(klazz: Class[_]): String = klazz.getSimpleName.stripSuffix("$")
  }

  /**
   * Test data directory
   * @return directory path
   */
  def testDataDir: String = {
    Some(new File("test-data")) filter (_.isDirectory) getOrElse new File("../test-data") getPath
  }

  /**
   * Load a file as string
   * @param path absolute or relative path of a file
   * @return the whole content of resource file as a string
   */
  def loadFile(path: String): String = {
    Source.fromFile(path).mkString
  }

  /**
   * Load a test resource file
   *
   * @param parent resource folder
   * @param name   resource name
   * @return resource file
   */
  def resourceFile(parent: String = resourceDir, name: String): File = {
    val file = new File(parent, name)
    if (!file.canRead) throw new IllegalStateException(s"File $file unreadable")
    file
  }

  /**
   * Load a test resource file as string
   *
   * @param parent   resource folder
   * @param noSpaces trim all spaces
   * @param name     resource name
   * @return resource file
   */
  @deprecated("Use loadResource", "3.2.3")
  def resourceString(parent: String = resourceDir, noSpaces: Boolean = true, name: String): String = {
    val file = resourceFile(parent = parent, name = name)
    val contents = Source.fromFile(file, "UTF-8").mkString
    if (noSpaces) contents.replaceAll("\\s", "") else contents
  }

  /**
   * Loads resource by path
   * @param path absolute or relative path of a resource
   * @return the whole content of resource file as a string
   */
  def loadResource(path: String, noSpaces: Boolean = false): String = {
    val raw = Source.fromInputStream(getClass.getResourceAsStream(path)).getLines
    if (noSpaces) raw.mkString("").replaceAll("\\s", "") else raw.mkString("\n")
  }

}
