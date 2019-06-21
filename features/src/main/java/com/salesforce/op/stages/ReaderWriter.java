package com.salesforce.op.stages;

import java.lang.annotation.*;

/**
 * Stage class annotation to specify custom reader/writer implementation of [[OpPipelineStageReaderWriter]].
 * Reader/writer implementation must extend [[OpPipelineStageReaderWriter]] trait
 * and has a single no arguments constructor.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@Inherited
public @interface ReaderWriter {

    /**
     * Reader/writer class extending [[OpPipelineStageReaderWriter]] to use when reading/writing the stage.
     * It must extend [[OpPipelineStageReaderWriter]] trait and has a single no arguments constructor.
     */
    Class<?> value();

}
