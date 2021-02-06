<template>
  <div>
    <div
      class="thumbnail-wrapper has-text-centered"
      v-on:drop.prevent="addImage"
      v-on:dragover.prevent
      v-on:dragenter="dragover++"
      v-on:dragleave="dragover--"
      v-bind:class="{ hover: dragover }"
    >
      <input v-bind:id="id" v-on:change="addImage" class="image-input" type="file" accept="image/*" />
      <label v-bind:for="id" class="image-input-label is-size-5">
        <strong>Choose a file</strong><span> or drag it here</span>
      </label>
      <figure class="image">
        <img v-bind:src="thumbnail" v-bind:style="cssVars" class="thumbnail" />
      </figure>
    </div>
    <div v-if="file">
      File name: <span v-if="file">{{ file.name }} ({{ fileSize }} KiB)</span>
    </div>
    <div v-else>File not selected</div>
  </div>
</template>

<script>
export default {
  components: {},

  data() {
    return {
      dragover: 0,
      file: null,
      thumbnail: '',
    };
  },

  props: {
    id: {
      type: String,
      required: true,
    },
    value: {
      type: File,
      default: null,
    },
    thumbnailSize: {
      type: Number,
      default: 256,
    },
  },

  computed: {
    cssVars() {
      return {
        '--thumbnail-size': `${this.thumbnailSize}px`,
      };
    },

    fileSize() {
      const sizeKb = this.file.size / 1024;
      return sizeKb.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    },
  },

  methods: {
    addImage(event) {
      // Handle both input and drop events
      const files = event.target.files || event.dataTransfer.files;

      this.dragover = 0;
      if (files.length === 0) {
        return;
      }

      const reader = new FileReader();
      reader.onload = (readerEvent) => {
        this.thumbnail = readerEvent.target.result;
      };

      [this.file] = files;
      reader.readAsDataURL(this.file);
      this.$emit('input', this.file);
    },
  },
};
</script>

<style scoped lang="scss">
.thumbnail-wrapper {
  padding: 2rem;
  outline: 2px dashed grey;
  outline-offset: -1rem;
}

.image-input {
  position: absolute;
  z-index: -1;
  width: 0.01px;
  height: 0.01px;
  opacity: 0;
}

.image-input-label {
  max-width: 80%;
  cursor: pointer;
}

.thumbnail {
  width: auto;
  height: var(--thumbnail-size);
  margin-right: auto;
  margin-left: auto;
}

.hover {
  background-color: lightgrey;
  opacity: 0.5;
}
</style>
