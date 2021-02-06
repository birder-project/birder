<template>
  <div>
    <Navbar />
    <div class="container">
      <div class="columns">
        <div class="column is-8 is-offset-2 is-centered">
          <h2 class="title is-2 has-text-centered">Birder</h2>
          <h3 class="subtitle is-3 has-text-centered">API Showcase</h3>
          <div class="columns">
            <div class="column">
              Select model
              <div class="select">
                <select v-model="selectedModel">
                  <template v-for="model in models">
                    <option v-bind:key="model" v-bind:value="model">{{ model }}</option>
                  </template>
                </select>
              </div>
            </div>
          </div>
          <div class="columns">
            <div class="column">
              <Language></Language>
            </div>
          </div>
          <div class="columns">
            <div class="column">
              <ImageUpload v-model="imageFile" id="image-upload"></ImageUpload>
            </div>
          </div>
          <div class="columns">
            <div class="column">
              <button v-on:click="submit()" class="button is-link">Submit</button>
            </div>
          </div>
          <div v-if="errorMessage" class="columns">
            <div class="column">
              <div class="has-text-danger">{{ errorMessage }}</div>
            </div>
          </div>
          <div class="columns">
            <div class="column">
              <template v-for="prediction in predictions">
                <div v-bind:key="prediction.class">
                  {{ prediction.class }} - {{ probabilityText(prediction.probability) }}%
                </div>
              </template>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import Navbar from '@/components/TheNavbar.vue';
import Language from '@/components/Language.vue';
import ImageUpload from '@/components/ImageUpload.vue';
import mmsManagementMutationType from '@/store/modules/mmsManagement/mutation-types';

export default {
  components: {
    Navbar,
    Language,
    ImageUpload,
  },

  data() {
    return {
      imageFile: null,
      predictions: {},
      errorMessage: null,
    };
  },

  computed: {
    models() {
      return this.$store.state.mmsManagement.models;
    },

    selectedModel: {
      get() {
        return this.$store.state.mmsManagement.selectedModel;
      },

      set(newValue) {
        this.$store.commit(mmsManagementMutationType.setModel, newValue);
      },
    },
  },

  watch: {
    models: {
      handler(newValue, _oldValue) {
        if (newValue.length === 0) {
          this.selectedModel = null;
        } else if (this.selectedModel === null) {
          [this.selectedModel] = newValue;
        } else if (newValue.includes(this.selectedModel) !== true) {
          [this.selectedModel] = newValue;
        }
      },
    },
  },

  methods: {
    submit() {
      const formData = new FormData();
      formData.append('image', this.imageFile);
      formData.append('model', this.selectedModel);

      axios
        .post('/predict/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            'Accept-Language': this.$store.state.language.selectedLanguage,
          },
        })
        .then((response) => {
          this.predictions = response.data;
          this.errorMessage = null;
        })
        .catch((error) => {
          console.error(error);
          if (error.isAxiosError === true && error.code === 'ECONNABORTED') {
            this.errorMessage = error.message;
          } else {
            this.errorMessage = `${error.response.status}: ${error.response.statusText}`;
          }
        });
    },

    probabilityText(probability) {
      const percent = probability * 100;
      return percent.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    },
  },
};
</script>
