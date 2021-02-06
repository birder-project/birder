<template>
  <div>
    Select language
    <div class="select">
      <select v-model="selectedLanguage">
        <template v-for="(language, code) in languages">
          <option v-bind:key="code" v-bind:value="code">{{ language }}</option>
        </template>
      </select>
    </div>
  </div>
</template>

<script>
import languageActionType from '@/store/modules/language/action-types';
import languageMutationType from '@/store/modules/language/mutation-types';

export default {
  components: {},

  data() {
    return {};
  },

  computed: {
    languages() {
      return this.$store.state.language.languages;
    },

    selectedLanguage: {
      get() {
        return this.$store.state.language.selectedLanguage;
      },

      set(newValue) {
        this.$store.commit(languageMutationType.setLanguage, newValue);
      },
    },
  },

  watch: {
    languages: {
      handler(newValue, _oldValue) {
        if (newValue.length === 0) {
          this.selectedLanguage = null;
        } else if (this.selectedLanguage === null) {
          [this.selectedLanguage] = Object.keys(newValue);
        } else if (Object.keys(newValue).includes(this.selectedLanguage) !== true) {
          [this.selectedLanguage] = Object.keys(newValue);
        }
      },
    },
  },

  created() {
    this.$store.dispatch(languageActionType.getLanguages);
  },
};
</script>
